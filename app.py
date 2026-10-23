# app.py
# Coconut Production Analytics Dashboard
# Run locally: streamlit run app.py
# Streamlit Community Cloud: keep app.py and requirements.txt in the repository root.
# Users upload their own CSV; no dataset needs to be stored in GitHub.

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyRegressor

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

st.set_page_config(
    page_title="Coconut Production Analytics",
    page_icon="🥥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
.dashboard-title {font-size: 2.15rem; font-weight: 750; letter-spacing: -0.04em;}
.muted {color: #737783;}
div[data-testid="stMetric"] {
    background: #f7f8fb; border: 1px solid #e7e9ef;
    padding: 13px 15px; border-radius: 12px;
}
section[data-testid="stSidebar"] {border-right: 1px solid #e7e9ef;}
</style>
""", unsafe_allow_html=True)

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
MONTH_MAP = {month.lower(): i for i, month in enumerate(MONTHS, start=1)}

NUMERIC_HINTS = [
    "year", "harvest", "production", "yield", "area", "price",
    "rainfall", "temperature", "inflation", "supply", "volume",
    "quantity", "amount", "ton", "kg", "hectare", "humidity"
]


def load_csv(uploaded_file):
    """Read CSV files with common encodings, including Windows-encoded files."""
    encodings = ["utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Unable to decode this CSV file. Details: {last_error}")


def clean_dataset(raw, trim_text=True, remove_duplicates=True):
    data = raw.copy()
    data.columns = [str(c).strip() for c in data.columns]

    # Ensure column names are unique.
    seen, new_columns = {}, []
    for col in data.columns:
        seen[col] = seen.get(col, 0) + 1
        new_columns.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
    data.columns = new_columns

    data = data.dropna(how="all").dropna(axis=1, how="all")

    if trim_text:
        for col in data.select_dtypes(include=["object", "string"]).columns:
            data[col] = data[col].astype("string").str.strip()
            data[col] = data[col].replace({
                "": pd.NA, "NA": pd.NA, "N/A": pd.NA,
                "null": pd.NA, "None": pd.NA, "nan": pd.NA
            })

    for col in data.columns:
        name = str(col).lower()
        if any(hint in name for hint in NUMERIC_HINTS):
            original_nonmissing = int(data[col].notna().sum())
            converted = pd.to_numeric(
                data[col].astype("string")
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("₱", "", regex=False)
                .str.strip(),
                errors="coerce"
            )
            if original_nonmissing > 0 and converted.notna().sum() >= max(
                1, int(original_nonmissing * 0.5)
            ):
                data[col] = converted

    duplicate_count = int(data.duplicated().sum())
    if remove_duplicates:
        data = data.drop_duplicates().copy()

    # Treat negative values in nonnegative measures as invalid/missing.
    nonnegative_tokens = [
        "harvest", "production", "yield", "area", "rainfall",
        "price", "supply", "volume", "quantity"
    ]
    for col in data.columns:
        name = str(col).lower()
        if (pd.api.types.is_numeric_dtype(data[col])
                and any(token in name for token in nonnegative_tokens)):
            data.loc[data[col] < 0, col] = np.nan

    return data, duplicate_count


def get_numeric_columns(data):
    return data.select_dtypes(include=np.number).columns.tolist()


def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]),
            numeric
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical
        ))
    if not transformers:
        raise ValueError("No usable predictor columns were found.")
    return ColumnTransformer(transformers, remainder="drop")


def plot_histogram(series, title, xlabel):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        st.info("There are no valid numeric values to plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=25, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Records")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_correlation(data, columns):
    corr = data[columns].corr(numeric_only=True)
    if corr.empty:
        st.info("Not enough numeric data to calculate correlations.")
        return
    fig, ax = plt.subplots(figsize=(max(7, len(columns) * 0.8), 5.5))
    image = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=65, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Correlation between numeric variables")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def quality_report(data):
    return pd.DataFrame({
        "Column": data.columns,
        "Data type": [str(data[c].dtype) for c in data.columns],
        "Missing values": [int(data[c].isna().sum()) for c in data.columns],
        "Missing (%)": [(data[c].isna().mean() * 100).round(2) for c in data.columns],
        "Unique values": [int(data[c].nunique(dropna=True)) for c in data.columns],
    })


# ---------- Sidebar ----------
st.sidebar.markdown("## 🥥 Coconut Analytics")
st.sidebar.caption("Production • Statistics • Prediction")
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"],
    help="Upload a CSV or Excel file from your device. The dataset does not need to be in GitHub."
)

with st.sidebar.expander("Cleaning settings", expanded=False):
    trim_text = st.checkbox("Trim text and standardize blanks", value=True)
    remove_duplicates = st.checkbox("Remove exact duplicate rows", value=True)
    preview_rows = st.selectbox("Preview rows", [5, 10, 20, 50], index=1)

page = st.sidebar.radio(
    "Dashboard navigation",
    [
        "Overview",
        "Data Cleaning",
        "Statistics & Exploration",
        "Classification & Prediction",
        "Harvest Forecasting",
        "Downloads",
        "Coconut Supply & Inflation",
    ],
)

st.markdown(
    '<div class="dashboard-title">Coconut Production Analytics</div>',
    unsafe_allow_html=True
)
st.caption(
    "Analyze coconut production records, explore trends, evaluate prediction models, "
    "and forecast future harvest."
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel dataset using the left sidebar to begin. No dataset is loaded from GitHub.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Upload")
        st.write("Choose a dataset from your computer.")
    with col2:
        st.markdown("#### 2. Explore")
        st.write("Review missing values, statistics, and production trends.")
    with col3:
        st.markdown("#### 3. Analyze")
        st.write("Run prediction models, forecasting, and export results.")
    st.markdown(
        "**Recommended fields (your column names may differ):** Year, Month, "
        "Barangay, Harvest (kg), Harvested Area (ha), Rainfall, Temperature, and prices."
    )
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        raw_data = load_csv(uploaded_file)
    else:
        raw_data = pd.read_excel(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded dataset: {exc}")
    st.stop()

if raw_data.empty:
    st.error("The uploaded file contains column headers but no data rows.")
    st.stop()

df, duplicates_found = clean_dataset(raw_data, trim_text, remove_duplicates)
if df.empty:
    st.error("No records remain after cleaning. Review the uploaded file and cleaning settings.")
    st.stop()

for key in [
    "forecast_download", "prediction_download", "classification_prediction_download"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------- Overview ----------
if page == "Overview":
    st.markdown("### Dataset overview")
    a, b, c, d = st.columns(4)
    a.metric("Records after cleaning", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Duplicates found", f"{duplicates_found:,}")
    d.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.markdown("### Uploaded data preview")
    st.dataframe(df.head(preview_rows), width="stretch")

    st.markdown("### Data quality report")
    st.dataframe(quality_report(df), width="stretch", hide_index=True)
    st.caption(
        "Exact duplicates are removed only if enabled. Negative values in production-like "
        "measures are treated as missing, not as valid observations."
    )

# ---------- Data Cleaning ----------
elif page == "Data Cleaning":
    st.markdown("### Data cleaning")
    st.write("Review missing values, inspect duplicates, choose an imputation method, and download a cleaned copy.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Exact duplicates found", f"{duplicates_found:,}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.markdown("#### Missing-value report")
    st.dataframe(
        quality_report(df).sort_values("Missing (%)", ascending=False),
        width="stretch", hide_index=True
    )
    fill_method = st.selectbox(
        "How should remaining missing values be handled?",
        [
            "Leave missing values as-is",
            "Numeric: median; text: most frequent",
            "Numeric: mean; text: most frequent",
            "Drop rows with any missing values"
        ],
        key="clean_fill_method"
    )
    cleaned_output = df.copy()
    if fill_method in (
        "Numeric: median; text: most frequent",
        "Numeric: mean; text: most frequent"
    ):
        for col in cleaned_output.columns:
            if pd.api.types.is_numeric_dtype(cleaned_output[col]):
                if cleaned_output[col].notna().any():
                    value = (
                        cleaned_output[col].median()
                        if fill_method.startswith("Numeric: median")
                        else cleaned_output[col].mean()
                    )
                    cleaned_output[col] = cleaned_output[col].fillna(value)
            elif cleaned_output[col].notna().any():
                mode = cleaned_output[col].mode(dropna=True)
                if not mode.empty:
                    cleaned_output[col] = cleaned_output[col].fillna(mode.iloc[0])
    elif fill_method == "Drop rows with any missing values":
        cleaned_output = cleaned_output.dropna().copy()

    st.markdown("#### Cleaned data preview")
    st.write(f"Rows in downloadable cleaned output: **{len(cleaned_output):,}**")
    st.dataframe(cleaned_output.head(preview_rows), width="stretch")
    st.download_button(
        "Download cleaned dataset (CSV)",
        data=cleaned_output.to_csv(index=False).encode("utf-8-sig"),
        file_name="cleaned_coconut_data.csv",
        mime="text/csv",
        width="stretch"
    )

# ---------- Statistics / EDA ----------
elif page == "Statistics & Exploration":
    numeric_cols = get_numeric_columns(df)
    if numeric_cols:
        st.markdown("### Descriptive statistics")
        st.dataframe(df[numeric_cols].describe().T.round(3), width="stretch")

        st.markdown("### Correlation analysis")
        if len(numeric_cols) >= 2:
            plot_correlation(df, numeric_cols)
        else:
            st.info("At least two numeric columns are needed for correlation analysis.")

        selected_num = st.selectbox("Select a numeric field to explore", numeric_cols)
        plot_histogram(df[selected_num], f"Distribution of {selected_num}", selected_num)
    else:
        st.warning("No numeric columns were detected. Check whether numeric fields contain text or symbols.")

    possible_categories = [
        col for col in df.columns
        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        )
        and 1 < df[col].nunique(dropna=True) <= 100
    ]
    if possible_categories:
        st.markdown("### Category breakdown")
        category = st.selectbox("Choose a category field", possible_categories)
        counts = df[category].fillna("Missing").astype(str).value_counts().head(20)
        st.dataframe(counts.rename("Records").to_frame(), width="stretch")
        fig, ax = plt.subplots(figsize=(9, 4))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Records by {category}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    barangay_col = next(
        (c for c in df.columns if c.lower().strip() in ("barangay", "village", "location")),
        None
    )
    harvest_col = next(
        (c for c in df.columns if "harvest" in c.lower() and pd.api.types.is_numeric_dtype(df[c])),
        None
    )
    if barangay_col and harvest_col:
        st.markdown("### Harvest by location")
        grouped = df.groupby(barangay_col, dropna=False)[harvest_col].sum(min_count=1).sort_values(ascending=False)
        st.dataframe(grouped.rename(f"Total {harvest_col}").to_frame(), width="stretch")
        fig, ax = plt.subplots(figsize=(10, 4))
        grouped.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel(harvest_col)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---------- Classification / Prediction ----------
elif page == "Coconut Supply & Inflation":
    st.markdown("### Annual Coconut Supply and Local Inflation Analysis")
    st.write(
        "This analysis aggregates the combined barangay dataset into one record per year. "
        "It evaluates annual inflation using a chronological holdout so later years are not "
        "randomly mixed into training data. Results describe predictive performance and do not establish causation."
    )

    required = ["Year", "Inflation Rate (%)"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        st.warning(
            "Annual mode requires these columns: " + ", ".join(missing_required) +
            ". Upload the combined coconut dataset or rename the matching columns."
        )
    else:
        raw = df.copy()
        raw["Year"] = pd.to_numeric(raw["Year"], errors="coerce")
        raw["Inflation Rate (%)"] = pd.to_numeric(raw["Inflation Rate (%)"], errors="coerce")
        raw = raw.dropna(subset=["Year", "Inflation Rate (%)"]).copy()
        raw["Year"] = raw["Year"].astype(int)

        # Aggregate monthly/barangay records into annual values.
        # Harvest is summed across months and barangays; rainfall is annual total;
        # prices and temperature are annual averages. Area is counted once per
        # barangay-year (not once per repeated month), then summed across barangays.
        agg = {}
        for col in ["Harvest (kg)", "Rainfall (mm)"]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
                agg[col] = "sum"
        for col in ["Farmgate Price (PHP/kg)", "Copra Price (PHP/kg)",
                    "Whole Nut Price (PHP/kg)", "Temperature (°C)"]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
                agg[col] = "mean"

        annual = raw.groupby("Year", as_index=False).agg({"Inflation Rate (%)": "mean"})
        if "Harvested Area (ha)" in raw.columns:
            area = raw[["Year", "Barangay", "Harvested Area (ha)"]].copy() if "Barangay" in raw.columns else raw[["Year", "Harvested Area (ha)"]].copy()
            area["Harvested Area (ha)"] = pd.to_numeric(area["Harvested Area (ha)"], errors="coerce")
            area = area.drop_duplicates(subset=["Year", "Barangay"] if "Barangay" in area.columns else ["Year"])
            area_year = area.groupby("Year", as_index=False)["Harvested Area (ha)"].sum(min_count=1)
            annual = annual.merge(area_year, on="Year", how="left")
        if agg:
            measures = raw.groupby("Year", as_index=False).agg(agg)
            annual = annual.merge(measures, on="Year", how="left")

        annual = annual.sort_values("Year").reset_index(drop=True)
        st.markdown("#### Annual dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual observations", len(annual))
        c2.metric("First year", int(annual["Year"].min()) if len(annual) else "—")
        c3.metric("Last year", int(annual["Year"].max()) if len(annual) else "—")
        st.caption(
            "Aggregation: harvest and rainfall are summed by year; harvested area is counted once per barangay-year and summed; "
            "prices, temperature, and inflation are averaged. Check the source documentation to confirm these units and definitions."
        )
        st.dataframe(annual, width="stretch", hide_index=True)

        candidate_predictors = [
            c for c in [
                "Harvested Area (ha)", "Harvest (kg)", "Farmgate Price (PHP/kg)",
                "Copra Price (PHP/kg)", "Whole Nut Price (PHP/kg)",
                "Rainfall (mm)", "Temperature (°C)"
            ] if c in annual.columns and annual[c].notna().sum() >= 3 and annual[c].nunique(dropna=True) > 1
        ]
        selected_predictors = st.multiselect(
            "Annual predictors",
            options=candidate_predictors,
            default=candidate_predictors,
            help="Predictors are annual summaries from the selected year. This measures same-year association/prediction, not a future-year forecast."
        )
        holdout_years = st.slider(
            "Number of final years reserved for testing",
            min_value=3, max_value=min(8, max(3, len(annual) - 8)),
            value=min(6, max(3, len(annual) - 8)),
            help="The final consecutive years are held out for testing. All earlier years are used for training."
        ) if len(annual) >= 12 else 3

        if len(selected_predictors) == 0:
            st.info("Select at least one annual predictor to compare the regression models.")
        elif len(annual) < 12:
            st.warning("At least 12 annual observations are recommended. Results from a smaller series will be highly uncertain.")
        elif len(annual) <= holdout_years + 5:
            st.warning("There are too few annual observations for this holdout size. Reduce the number of test years.")
        else:
            # Do not drop missing predictor rows prematurely: the preprocessing pipeline
            # learns imputation values on training data only.
            model_data = annual[["Year", "Inflation Rate (%)"] + selected_predictors].copy()
            model_data = model_data.dropna(subset=["Inflation Rate (%)"]).sort_values("Year").reset_index(drop=True)
            split = len(model_data) - holdout_years
            train = model_data.iloc[:split].copy()
            test = model_data.iloc[split:].copy()
            X_train, X_test = train[selected_predictors], test[selected_predictors]
            y_train, y_test = train["Inflation Rate (%)"], test["Inflation Rate (%)"]
            st.info(
                f"Chronological holdout: training on {int(train['Year'].min())}–{int(train['Year'].max())} "
                f"({len(train)} years), testing on {int(test['Year'].min())}–{int(test['Year'].max())} "
                f"({len(test)} years). Preprocessing is fitted on training data only."
            )

            models = {
                "Multiple Linear Regression": Pipeline([
                    ("preprocess", make_preprocessor(X_train)),
                    ("model", LinearRegression())
                ]),
                "SVR": Pipeline([
                    ("preprocess", make_preprocessor(X_train)),
                    ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1))
                ]),
                "Deep Learning (ANN)": Pipeline([
                    ("preprocess", make_preprocessor(X_train)),
                    ("model", MLPRegressor(
                        hidden_layer_sizes=(8, 4), activation="relu", solver="adam",
                        alpha=0.05, learning_rate_init=0.005, max_iter=2000,
                        early_stopping=False, random_state=42
                    ))
                ])
            }

            rows, fitted_models, predictions = [], {}, {}
            def add_metrics(name, actual, predicted):
                actual = np.asarray(actual, dtype=float)
                predicted = np.asarray(predicted, dtype=float)
                mse = mean_squared_error(actual, predicted)
                rows.append({
                    "Model": name,
                    "MSE": mse,
                    "RMSE": float(np.sqrt(mse)),
                    "MAE": mean_absolute_error(actual, predicted),
                    "R²": r2_score(actual, predicted) if len(actual) > 1 else np.nan
                })
                predictions[name] = predicted

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    fitted_models[name] = model
                    add_metrics(name, y_test, pred)
                except Exception as exc:
                    st.warning(f"{name} could not be fitted: {exc}")

            # Baselines are evaluated on exactly the same test years.
            mean_pred = np.repeat(float(y_train.mean()), len(y_test))
            add_metrics("Baseline: training mean", y_test, mean_pred)
            naive_pred = np.repeat(float(y_train.iloc[-1]), len(y_test))
            add_metrics("Baseline: last training year", y_test, naive_pred)

            if HAS_STATSMODELS and len(y_train) >= 8:
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    arima = ARIMA(y_train.astype(float).to_numpy(), order=(1, 1, 1)).fit()
                    arima_pred = np.asarray(arima.forecast(steps=len(y_test)), dtype=float)
                    add_metrics("ARIMA (1,1,1)", y_test, arima_pred)
                except Exception as exc:
                    st.warning(f"ARIMA was not included: {exc}")
            else:
                st.caption("ARIMA requires statsmodels and at least 8 training years.")

            results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
            st.markdown("#### Model comparison — same held-out years")
            st.dataframe(results.round(4), width="stretch", hide_index=True)
            st.caption(
                "Lower MSE, RMSE, and MAE are better. Higher R² is better, but negative R² means the model performed worse "
                "than predicting the mean of the test targets. R² is not an accuracy percentage. Small annual samples can make metrics unstable."
            )

            eligible = results[results["Model"].isin(list(models.keys()) + ["ARIMA (1,1,1)"])]
            if not eligible.empty:
                best_name = str(eligible.sort_values("RMSE").iloc[0]["Model"])
                best_rmse = float(eligible.sort_values("RMSE").iloc[0]["RMSE"])
                st.metric("Lowest-RMSE fitted model", best_name, f"RMSE {best_rmse:.4f}")
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(test["Year"], y_test.to_numpy(), marker="o", label="Actual annual inflation")
                if best_name in predictions:
                    ax.plot(test["Year"], predictions[best_name], marker="o", label=f"Predicted — {best_name}")
                ax.set_xlabel("Year")
                ax.set_ylabel("Inflation Rate (%)")
                ax.set_title("Actual vs predicted annual inflation")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with st.expander("Test-set predictions"):
                pred_table = pd.DataFrame({
                    "Year": test["Year"].to_numpy(),
                    "Actual inflation (%)": y_test.to_numpy()
                })
                for name, pred in predictions.items():
                    pred_table[f"{name} prediction"] = pred
                st.dataframe(pred_table.round(4), width="stretch", hide_index=True)
                st.download_button(
                    "Download annual model predictions (CSV)",
                    data=pred_table.to_csv(index=False).encode("utf-8"),
                    file_name="annual_inflation_model_predictions.csv",
                    mime="text/csv"
                )

            st.markdown("#### Time-series cross-validation on training years")
            if len(train) >= 10:
                cv_splits = min(4, max(2, len(train) // 5))
                cv = TimeSeriesSplit(n_splits=cv_splits)
                cv_rows = []
                for name, model in models.items():
                    try:
                        fold_mae = []
                        for tr_idx, val_idx in cv.split(X_train):
                            fold_model = __import__("sklearn.base", fromlist=["clone"]).clone(model)
                            fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                            fold_pred = fold_model.predict(X_train.iloc[val_idx])
                            fold_mae.append(mean_absolute_error(y_train.iloc[val_idx], fold_pred))
                        cv_rows.append({
                            "Model": name,
                            "CV MAE mean": float(np.mean(fold_mae)),
                            "CV MAE std": float(np.std(fold_mae, ddof=0))
                        })
                    except Exception as exc:
                        cv_rows.append({"Model": name, "CV MAE mean": np.nan, "CV MAE std": np.nan, "Note": str(exc)})
                st.dataframe(pd.DataFrame(cv_rows).round(4), width="stretch", hide_index=True)
                st.caption("Cross-validation uses only training years; the final test years remain untouched.")
            else:
                st.caption("Not enough training years for reliable time-series cross-validation.")

            st.markdown("#### Annual inflation forecast")
            st.caption("Future-year forecasts are uncertain and should not be interpreted as official inflation estimates.")
            if HAS_STATSMODELS and len(model_data) >= 12:
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    full_series = model_data.sort_values("Year")["Inflation Rate (%)"].astype(float).to_numpy()
                    horizon = st.slider("Future annual forecast horizon", 1, 5, 3, key="annual_inflation_horizon")
                    order_p = st.selectbox("ARIMA p", [0, 1, 2], index=1, key="annual_infl_arima_p")
                    order_d = st.selectbox("ARIMA d", [0, 1], index=1, key="annual_infl_arima_d")
                    order_q = st.selectbox("ARIMA q", [0, 1, 2], index=1, key="annual_infl_arima_q")
                    forecast_model = ARIMA(full_series, order=(order_p, order_d, order_q)).fit()
                    forecast = np.asarray(forecast_model.forecast(steps=horizon), dtype=float)
                    last_year = int(model_data["Year"].max())
                    forecast_df = pd.DataFrame({
                        "Year": range(last_year + 1, last_year + horizon + 1),
                        "Forecast inflation (%)": forecast
                    })
                    st.dataframe(forecast_df.round(4), width="stretch", hide_index=True)
                except Exception as exc:
                    st.info(f"Forecast could not be calculated: {exc}")

elif page == "Classification & Prediction":
    numeric_cols = get_numeric_columns(df)
    suggested_targets = [
        c for c in numeric_cols
        if any(term in c.lower() for term in ("harvest", "production", "yield", "supply", "quantity", "volume"))
    ]
    if not suggested_targets:
        st.warning("No likely numeric production target was detected. Ensure your CSV has a numeric harvest, production, yield, supply, or quantity column.")
    else:
        # Automatically choose the most suitable overall production/supply measure.
        # Priority favors explicit supply/production quantity fields over generic harvest fields.
        target_priority = ("supply", "production", "harvest", "yield", "quantity", "volume")
        target = min(
            suggested_targets,
            key=lambda col: min(
                (target_priority.index(term) for term in target_priority if term in col.lower()),
                default=len(target_priority)
            )
        )
        st.caption(f"Automatic outcome selected from your data: **{target}**. No outcome-variable selection is required.")
        st.caption(
            "Models use the uploaded dataset only. Results are estimates and do not prove "
            "that a variable causes a change in production. The classification automatically "
            "groups the selected overall production/supply measure into Low, Medium, or High."
        )
        model_data = df[df[target].notna()].copy()
        feature_cols = []
        for col in model_data.columns:
            if col == target:
                continue
            if model_data[col].isna().all() or model_data[col].nunique(dropna=True) <= 1:
                continue
            if (
                (col.lower().strip() == "id" or col.lower().strip().endswith("_id"))
                and model_data[col].nunique() > 0.8 * len(model_data)
            ):
                continue
            feature_cols.append(col)

        if len(model_data) < 10:
            st.warning("At least 10 valid records are recommended for a basic train/test evaluation.")
        elif not feature_cols:
            st.warning("No usable predictor columns were found apart from the selected target.")
        else:
            X = model_data[feature_cols].copy()
            y = pd.to_numeric(model_data[target], errors="coerce")
            valid = y.notna()
            X, y = X.loc[valid], y.loc[valid]
            if len(X) < 10:
                st.warning("There are fewer than 10 valid target records after conversion.")
            else:
                st.markdown("### Regression: predict the selected quantity")
                test_fraction = st.slider("Test dataset proportion", 0.15, 0.40, 0.20, 0.05)
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_fraction, random_state=42
                    )
                    reg = Pipeline([
                        ("preprocess", make_preprocessor(X_train)),
                        ("model", MLPRegressor(
                            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                            alpha=0.001, learning_rate_init=0.001, max_iter=1000,
                            early_stopping=False, random_state=42
                        )),
                    ])
                    reg.fit(X_train, y_train)
                    prediction = reg.predict(X_test)
                    mae = mean_absolute_error(y_test, prediction)
                    rmse = mean_squared_error(y_test, prediction) ** 0.5
                    r2 = r2_score(y_test, prediction) if len(y_test) > 1 else np.nan
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE", f"{mae:,.3f}")
                    m2.metric("RMSE", f"{rmse:,.3f}")
                    m3.metric("R²", f"{r2:.3f}" if np.isfinite(r2) else "N/A")

                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(y_test, prediction, alpha=0.7)
                    low = min(float(y_test.min()), float(np.min(prediction)))
                    high = max(float(y_test.max()), float(np.max(prediction)))
                    if low == high:
                        pad = max(abs(low) * 0.05, 1.0)
                        low -= pad
                        high += pad
                    ax.plot([low, high], [low, high], linestyle="--")
                    ax.set_xlabel(f"Actual {target}")
                    ax.set_ylabel(f"Predicted {target}")
                    ax.set_title("Actual vs predicted")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    baseline = DummyRegressor(strategy="mean")
                    baseline.fit(X_train, y_train)
                    baseline_prediction = baseline.predict(X_test)
                    st.markdown("### Model comparison")
                    comparison = pd.DataFrame([
                        {"Model": "Deep Learning (ANN)", "MAE": mae, "RMSE": rmse, "R²": r2},
                        {
                            "Model": "Mean baseline",
                            "MAE": mean_absolute_error(y_test, baseline_prediction),
                            "RMSE": mean_squared_error(y_test, baseline_prediction) ** 0.5,
                            "R²": r2_score(y_test, baseline_prediction) if len(y_test) > 1 else np.nan,
                        }
                    ])
                    st.dataframe(comparison.round(4), width="stretch", hide_index=True)

                    st.markdown("### Predictor importance")
                    with st.spinner("Calculating permutation importance..."):
                        importance_result = permutation_importance(
                            reg, X_test, y_test, n_repeats=2, random_state=42,
                            scoring="neg_mean_absolute_error", n_jobs=1
                        )
                    importance = pd.DataFrame({
                        "Predictor": X_test.columns,
                        "Importance": importance_result.importances_mean
                    }).sort_values("Importance", ascending=False)
                    st.dataframe(importance, width="stretch", hide_index=True)
                    fig, ax = plt.subplots(figsize=(9, 4))
                    importance.sort_values("Importance").plot(
                        kind="barh", x="Predictor", y="Importance", legend=False, ax=ax
                    )
                    ax.set_title("Permutation importance (higher means more useful to this model)")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.session_state["prediction_download"] = pd.DataFrame({
                        "Actual": y_test.to_numpy(),
                        "Predicted": prediction
                    })
                except Exception as exc:
                    st.error(f"Regression could not run with the selected columns: {exc}")

                st.markdown("### Overall Coconut Supply Classification")
                st.write(
                    "The dashboard automatically classifies the selected production/supply measure "
                    "into Low, Medium, or High. You do not need to choose a separate classification "
                    "outcome or manually select input variables. Available non-target fields are used "
                    "together to evaluate combinations."
                )
                class_target = target
                raw_y = pd.to_numeric(model_data[class_target], errors="coerce")
                valid_numeric = raw_y.notna()
                y_class = pd.Series(pd.NA, index=model_data.index, dtype="string")
                if int(valid_numeric.sum()) >= 3 and raw_y.loc[valid_numeric].nunique() >= 2:
                    # Quantile groups produce relative Low / Medium / High supply classes.
                    # Rank first so tied values do not cause qcut to collapse bin edges.
                    try:
                        ranked = raw_y.loc[valid_numeric].rank(method="first")
                        binned = pd.qcut(ranked, q=3, labels=["Low", "Medium", "High"])
                        y_class.loc[valid_numeric] = binned.astype("string")
                    except (ValueError, TypeError):
                        try:
                            ranked = raw_y.loc[valid_numeric].rank(method="first")
                            binned = pd.qcut(ranked, q=2, labels=["Low", "High"])
                            y_class.loc[valid_numeric] = binned.astype("string")
                        except (ValueError, TypeError):
                            pass

                # Automatically use all usable columns except the classification target and
                # the target already used by regression. This avoids asking users to choose
                # a class outcome or individual predictor fields.
                selected_features = [
                    c for c in model_data.columns
                    if c != class_target
                    and model_data[c].notna().any()
                    and model_data[c].nunique(dropna=True) > 1
                    and not (
                        (c.lower().strip() == "id" or c.lower().strip().endswith("_id"))
                        and model_data[c].nunique(dropna=True) > 0.8 * len(model_data)
                    )
                ]
                # Prevent direct target leakage if the regression target and class target differ.
                selected_features = [c for c in selected_features if c != target]

                st.caption(f"Automatic outcome: {class_target} grouped into Low / Medium / High.")
                st.caption(
                    "The model uses all eligible columns in the uploaded data as inputs. "
                    "Weather, month, or other fields are optional predictors only when present; "
                    "none is required as the classification outcome."
                )

                if not selected_features:
                    st.warning("No usable input variables were found apart from the supply outcome.")
                else:
                    X_class = model_data[selected_features].copy()
                    valid = y_class.notna()
                    X_class, y_class = X_class.loc[valid], y_class.loc[valid].astype(str)
                    counts = y_class.value_counts()
                    keep = counts[counts >= 2].index
                    mask = y_class.isin(keep)
                    X_class, y_class = X_class.loc[mask], y_class.loc[mask]

                    if len(y_class) < 6 or y_class.nunique() < 2:
                        st.warning(
                            "There are not enough examples in multiple supply classes to train "
                            "and evaluate the classifier. Upload more valid records if possible."
                        )
                    else:
                        test_size_cls = st.slider(
                            "Classification test proportion", .15, .40, .20, .05,
                            key="classification_test_fraction"
                        )
                        class_counts = y_class.value_counts()
                        n_classes = int(y_class.nunique())
                        n_test = int(np.ceil(len(y_class) * float(test_size_cls)))
                        n_train = len(y_class) - n_test
                        can_stratify = (
                            n_classes >= 2 and int(class_counts.min()) >= 2
                            and n_test >= n_classes and n_train >= n_classes
                        )
                        stratify = y_class if can_stratify else None
                        try:
                            Xtr, Xte, ytr, yte = train_test_split(
                                X_class, y_class, test_size=test_size_cls,
                                random_state=42, stratify=stratify
                            )
                            clf = Pipeline([
                                ("preprocess", make_preprocessor(Xtr)),
                                ("model", RandomForestClassifier(
                                    n_estimators=250, random_state=42,
                                    class_weight="balanced", min_samples_leaf=1, n_jobs=1
                                ))
                            ])
                            clf.fit(Xtr, ytr)
                            yhat = clf.predict(Xte)
                            score1, score2 = st.columns(2)
                            score1.metric("Accuracy", f"{accuracy_score(yte, yhat):.1%}")
                            score2.metric("Balanced accuracy", f"{balanced_accuracy_score(yte, yhat):.1%}")
                            report = pd.DataFrame(
                                classification_report(yte, yhat, output_dict=True, zero_division=0)
                            ).T
                            report.index.name = "Class"
                            st.markdown("#### Classification report")
                            st.dataframe(report.round(4), width="stretch")
                            with st.expander("What do the report columns mean?"):
                                st.write("**Precision:** how often predictions for a class are correct. **Recall:** how many actual examples of a class were found. **F1-score:** balance between precision and recall. **Support:** number of true test examples in each class.")
                                st.write("**Macro avg** treats classes equally; **weighted avg** accounts for each class's number of records. These scores depend on dataset size and class distribution.")

                            labels = list(clf.named_steps["model"].classes_)
                            cm = confusion_matrix(yte, yhat, labels=labels)
                            fig, ax = plt.subplots(figsize=(max(5, len(labels) * .9), 4.5))
                            im = ax.imshow(cm, aspect="auto")
                            ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
                            ax.set_yticks(range(len(labels)), labels)
                            ax.set_xlabel("Predicted class")
                            ax.set_ylabel("Actual class")
                            ax.set_title("Overall coconut supply — confusion matrix")
                            for i in range(cm.shape[0]):
                                for j in range(cm.shape[1]):
                                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
                            fig.colorbar(im, ax=ax, label="Records")
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                            try:
                                perm = permutation_importance(
                                    clf, Xte, yte, n_repeats=5, random_state=42,
                                    scoring="f1_weighted", n_jobs=1
                                )
                                importance = pd.DataFrame({
                                    "Variable": selected_features,
                                    "Importance": perm.importances_mean,
                                    "Description": [
                                        f"Model usefulness of {f}; this is not evidence of causation."
                                        for f in selected_features
                                    ]
                                }).sort_values("Importance", ascending=False)
                                st.markdown("#### Automatically included input variable importance")
                                try:
                                    import plotly.express as px
                                    fig = px.bar(
                                        importance, x="Importance", y="Variable", orientation="h",
                                        hover_data={"Description": True, "Importance": ":.4f"},
                                        title="Which available inputs help classify overall supply?"
                                    )
                                    st.plotly_chart(fig, width="stretch")
                                except ImportError:
                                    st.bar_chart(importance.set_index("Variable")["Importance"], horizontal=True)
                                st.caption("Importance measures model usefulness, not causation.")
                            except Exception as imp_exc:
                                st.info(f"Variable importance could not be calculated: {imp_exc}")

                            st.markdown("#### Predict overall supply for a new combination")
                            st.caption("Enter values for the available inputs below. The model returns one overall Low, Medium, or High supply class.")
                            user_values = {}
                            for feature in selected_features:
                                series = X_class[feature]
                                if pd.api.types.is_numeric_dtype(series):
                                    vals = pd.to_numeric(series, errors="coerce").dropna()
                                    default = float(vals.median()) if not vals.empty else 0.0
                                    low = float(vals.min()) if not vals.empty else 0.0
                                    high = float(vals.max()) if not vals.empty else 1.0
                                    if low == high:
                                        high = low + 1.0
                                    user_values[feature] = st.number_input(
                                        feature, min_value=low, max_value=high,
                                        value=min(max(default, low), high), key=f"overall_combo_{feature}"
                                    )
                                else:
                                    opts = series.dropna().astype(str).value_counts().index.tolist() or ["Unknown"]
                                    user_values[feature] = st.selectbox(
                                        feature, opts, key=f"overall_combo_cat_{feature}"
                                    )

                            if st.button("Classify overall coconut supply", type="primary"):
                                one_row = pd.DataFrame([user_values], columns=selected_features)
                                outcome = str(clf.predict(one_row)[0])
                                st.success(f"Overall coconut supply classification: {outcome}")
                                if hasattr(clf.named_steps["model"], "predict_proba"):
                                    probs = clf.predict_proba(one_row)[0]
                                    prob_df = pd.DataFrame({
                                        "Class": clf.named_steps["model"].classes_,
                                        "Estimated probability": probs
                                    }).sort_values("Estimated probability", ascending=False)
                                    st.dataframe(
                                        prob_df.style.format({"Estimated probability": "{:.1%}"}),
                                        width="stretch", hide_index=True
                                    )
                                st.session_state["classification_prediction_download"] = pd.DataFrame(
                                    [{**user_values, "Automatic outcome": class_target,
                                      "Overall supply class": outcome}]
                                )
                        except Exception as exc:
                            st.error(f"Overall supply classification could not run with these data: {exc}")

# ---------- Forecasting ----------
elif page == "Harvest Forecasting":
    st.markdown("### Monthly harvest forecast")
    year_col = next(
        (c for c in df.columns if c.lower().strip() in ("year", "harvest year", "production year")),
        None
    )
    month_col = next(
        (c for c in df.columns if c.lower().strip() in ("month", "harvest month", "production month")),
        None
    )
    target_options = [
        c for c in get_numeric_columns(df)
        if any(term in c.lower() for term in ("harvest", "production", "yield", "supply", "quantity", "volume"))
    ]

    if not year_col or not month_col or not target_options:
        st.warning("Forecasting needs Year, Month, and a numeric harvest/production column. Check your CSV headers.")
    else:
        target = st.selectbox("Production field to forecast", target_options, key="forecast_target")
        forecast_data = df[[year_col, month_col, target]].copy()
        forecast_data[year_col] = pd.to_numeric(forecast_data[year_col], errors="coerce")
        forecast_data[target] = pd.to_numeric(forecast_data[target], errors="coerce")
        month_values = forecast_data[month_col].astype("string").str.strip().str.lower()
        forecast_data["__month_num"] = month_values.map(MONTH_MAP)
        numeric_months = pd.to_numeric(month_values, errors="coerce")
        forecast_data["__month_num"] = forecast_data["__month_num"].fillna(numeric_months)
        forecast_data = forecast_data.dropna(subset=[year_col, "__month_num", target])
        forecast_data = forecast_data[
            forecast_data["__month_num"].between(1, 12)
            & forecast_data[year_col].between(1900, 2200)
        ].copy()

        if forecast_data.empty:
            st.warning("No valid year/month/production records were found. Month can be a full month name or a number from 1 to 12.")
        else:
            forecast_data["Date"] = pd.to_datetime({
                "year": forecast_data[year_col].astype(int),
                "month": forecast_data["__month_num"].astype(int),
                "day": 1,
            })
            monthly = forecast_data.groupby("Date")[target].sum().sort_index().asfreq("MS")
            missing_months = int(monthly.isna().sum())
            monthly = monthly.interpolate(method="time", limit_direction="both")

            st.metric("Monthly observations", f"{len(monthly):,}")
            st.caption(f"Missing calendar months before interpolation: {missing_months}. Interpolation is used for plotting/model continuity and may affect results.")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(monthly.index, monthly.values)
            ax.set_title("Historical monthly total")
            ax.set_ylabel(target)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if len(monthly) < 6:
                st.warning("At least 6 monthly observations are recommended for a basic forecast; more history is better.")
            else:
                horizon = st.slider("Months to forecast", 3, 24, 12)
                future_index = pd.date_range(
                    monthly.index[-1] + pd.offsets.MonthBegin(1),
                    periods=horizon, freq="MS"
                )
                forecast_values = None
                method = "Seasonal-naive fallback"
                if HAS_STATSMODELS and len(monthly) >= 24:
                    try:
                        fitted = ExponentialSmoothing(
                            monthly, trend="add", seasonal="add",
                            seasonal_periods=12, initialization_method="estimated"
                        ).fit(optimized=True)
                        forecast_values = np.asarray(fitted.forecast(horizon))
                        method = "Holt-Winters Exponential Smoothing"
                    except Exception:
                        forecast_values = None
                if forecast_values is None:
                    vals = []
                    for date in future_index:
                        prior_year_date = date - pd.DateOffset(months=12)
                        if prior_year_date in monthly.index and pd.notna(monthly.loc[prior_year_date]):
                            vals.append(float(monthly.loc[prior_year_date]))
                        else:
                            vals.append(float(monthly.tail(min(12, len(monthly))).median()))
                    forecast_values = np.asarray(vals)

                forecast_values = np.maximum(forecast_values, 0)
                future = pd.DataFrame({
                    "Date": future_index,
                    f"Forecast {target}": forecast_values,
                    "Method": method,
                })
                st.write(f"Forecast method: **{method}**")
                st.dataframe(future.round(2), width="stretch", hide_index=True)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(monthly.index, monthly.values, label="Historical")
                ax.plot(future_index, forecast_values, marker="o", label="Forecast")
                ax.set_title(f"Forecast for the next {horizon} months")
                ax.set_ylabel(target)
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.session_state["forecast_download"] = future

# ---------- Downloads ----------
elif page == "Downloads":
    st.markdown("### Download your analysis outputs")
    st.write("These files are generated from the dataset uploaded in your current session.")
    st.download_button(
        "Download cleaned dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="cleaned_coconut_data.csv",
        mime="text/csv",
        width="stretch",
    )

    st.download_button(
        "Download data quality report (CSV)",
        data=quality_report(df).to_csv(index=False).encode("utf-8-sig"),
        file_name="data_quality_report.csv",
        mime="text/csv",
        width="stretch",
    )

    forecast = st.session_state.get("forecast_download")
    if forecast is not None:
        st.download_button(
            "Download generated forecast (CSV)",
            data=forecast.to_csv(index=False).encode("utf-8-sig"),
            file_name="coconut_production_forecast.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("Open Harvest Forecasting and generate a forecast first to enable the forecast download.")

    prediction = st.session_state.get("prediction_download")
    if prediction is not None:
        st.download_button(
            "Download latest test predictions (CSV)",
            data=prediction.to_csv(index=False).encode("utf-8-sig"),
            file_name="model_test_predictions.csv",
            mime="text/csv",
            width="stretch",
        )

    class_prediction_download = st.session_state.get("classification_prediction_download")
    if class_prediction_download is not None:
        st.download_button(
            "Download latest combination prediction (CSV)",
            data=class_prediction_download.to_csv(index=False).encode("utf-8-sig"),
            file_name="combination_prediction.csv",
            mime="text/csv",
            width="stretch",
        )

st.markdown("---")
st.caption(
    "Coconut Production Analytics • Uploads are processed by this app for the current session. "
    "Model results depend on the quality, quantity, and coverage of the uploaded records."
)
