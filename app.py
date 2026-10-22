from pathlib import Path

code = r'''# app.py
# Coconut Production Analytics Dashboard
# Run locally: streamlit run app.py
# Deploy to Streamlit Community Cloud with app.py and requirements.txt only.
# Users upload their own CSV in the browser; no dataset needs to be stored in GitHub.

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

# ---------- Styling ----------
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

# ---------- Helpers ----------
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

def read_uploaded_csv(uploaded_file):
    """Read a user-uploaded CSV without requiring a repository dataset."""
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not read this CSV. Please check that it is a valid CSV file. " + " | ".join(errors[-2:]))

def clean_dataset(raw, trim_text=True, remove_duplicates=True):
    data = raw.copy()
    data.columns = [str(c).strip() for c in data.columns]
    # Make duplicate column names unique to avoid ambiguous pandas operations.
    seen = {}
    new_columns = []
    for col in data.columns:
        seen[col] = seen.get(col, 0) + 1
        new_columns.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
    data.columns = new_columns

    if trim_text:
        for col in data.select_dtypes(include=["object", "string"]).columns:
            data[col] = data[col].astype("string").str.strip()
            data[col] = data[col].replace({
                "": pd.NA, "NA": pd.NA, "N/A": pd.NA,
                "null": pd.NA, "None": pd.NA, "nan": pd.NA
            })

    # Convert likely numeric columns when their names suggest numeric values.
    for col in data.columns:
        name = str(col).lower()
        if any(hint in name for hint in NUMERIC_HINTS):
            original_nonmissing = data[col].notna().sum()
            converted = pd.to_numeric(
                data[col].astype("string").str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False),
                errors="coerce"
            )
            # Avoid turning categorical labels into all-missing columns.
            if original_nonmissing == 0 or converted.notna().sum() >= max(1, int(original_nonmissing * 0.5)):
                data[col] = converted

    duplicate_count = int(data.duplicated().sum())
    if remove_duplicates:
        data = data.drop_duplicates().copy()

    # Negative values are invalid for these numeric measures; mark as missing.
    for col in data.columns:
        name = str(col).lower()
        if pd.api.types.is_numeric_dtype(data[col]) and any(
            token in name for token in ["harvest", "production", "yield", "area", "rainfall", "price", "supply", "volume", "quantity"]
        ):
            data.loc[data[col] < 0, col] = np.nan
    return data, duplicate_count

def get_numeric_columns(data):
    return data.select_dtypes(include=np.number).columns.tolist()

def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
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

# ---------- Sidebar ----------
st.sidebar.markdown("## 🥥 Coconut Analytics")
st.sidebar.caption("Production • Statistics • Prediction")
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv"],
    help="Upload a CSV file from your device. The dataset does not need to be in GitHub."
)

with st.sidebar.expander("Cleaning settings", expanded=False):
    trim_text = st.checkbox("Trim text and standardize blanks", value=True)
    remove_duplicates = st.checkbox("Remove exact duplicate rows", value=True)
    preview_rows = st.selectbox("Preview rows", [5, 10, 20, 50], index=1)

page = st.sidebar.radio(
    "Dashboard navigation",
    [
        "Overview & Data Quality",
        "Statistics & Exploration",
        "Classification & Prediction",
        "Harvest Forecasting",
        "Downloads",
    ],
)

# ---------- Header ----------
st.markdown('<div class="dashboard-title">Coconut Production Analytics</div>', unsafe_allow_html=True)
st.caption("Analyze coconut production records, explore trends, evaluate prediction models, and forecast future harvest.")

if uploaded_file is None:
    st.info("Upload a CSV dataset using the left sidebar to begin. No dataset is loaded from GitHub.")
    st.markdown("### Getting started")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Upload")
        st.write("Choose a CSV file from your computer.")
    with col2:
        st.markdown("#### 2. Explore")
        st.write("Review missing values, statistics, and production trends.")
    with col3:
        st.markdown("#### 3. Analyze")
        st.write("Run classification, regression, forecasting, and export results.")
    st.markdown("**Recommended fields (your column names may differ):** Year, Month, Barangay, Harvest (kg), Harvested Area (ha), Rainfall, Temperature, and prices.")
    st.stop()

try:
    raw_data = read_uploaded_csv(uploaded_file)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if raw_data.empty:
    st.error("The CSV contains column headers but no data rows.")
    st.stop()

df, duplicates_found = clean_dataset(raw_data, trim_text, remove_duplicates)
if df.empty:
    st.error("No records remain after cleaning. Review the uploaded file and cleaning settings.")
    st.stop()

# Persist generated forecast within the current Streamlit session only.
if "forecast_download" not in st.session_state:
    st.session_state["forecast_download"] = None

# ---------- Overview ----------
if page == "Overview & Data Quality":
    st.markdown("### Dataset overview")
    a, b, c, d = st.columns(4)
    a.metric("Records after cleaning", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Duplicates found", f"{duplicates_found:,}")
    d.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.markdown("### Uploaded data preview")
    st.dataframe(df.head(preview_rows), use_container_width=True)

    st.markdown("### Data quality report")
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Missing (%)": [(df[c].isna().mean() * 100).round(2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.caption("Exact duplicates are removed only if enabled. Negative values in production-like measures are treated as missing, not as valid observations.")

# ---------- Statistics / EDA ----------
elif page == "Statistics & Exploration":
    numeric_cols = get_numeric_columns(df)
    if numeric_cols:
        st.markdown("### Descriptive statistics")
        st.dataframe(df[numeric_cols].describe().T.round(3), use_container_width=True)

        st.markdown("### Correlation analysis")
        if len(numeric_cols) >= 2:
            plot_correlation(df, numeric_cols)
        else:
            st.info("At least two numeric columns are needed for correlation analysis.")

        selected_num = st.selectbox("Select a numeric field to explore", numeric_cols)
        plot_histogram(df[selected_num], f"Distribution of {selected_num}", selected_num)
    else:
        st.warning("No numeric columns were detected. Check whether numeric fields in your CSV contain text or symbols.")

    possible_categories = [
        col for col in df.columns
        if (df[col].dtype == "object" or str(df[col].dtype) in ("string", "category", "bool"))
        and 1 < df[col].nunique(dropna=True) <= 100
    ]
    if possible_categories:
        st.markdown("### Category breakdown")
        category = st.selectbox("Choose a category field", possible_categories)
        counts = df[category].fillna("Missing").astype(str).value_counts().head(20)
        st.dataframe(counts.rename("Records").to_frame(), use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Records by {category}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    barangay_col = next((c for c in df.columns if c.lower().strip() in ("barangay", "village", "location")), None)
    harvest_col = next((c for c in df.columns if "harvest" in c.lower() and pd.api.types.is_numeric_dtype(df[c])), None)
    if barangay_col and harvest_col:
        st.markdown("### Harvest by location")
        grouped = df.groupby(barangay_col, dropna=False)[harvest_col].sum(min_count=1).sort_values(ascending=False)
        st.dataframe(grouped.rename(f"Total {harvest_col}").to_frame(), use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        grouped.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel(harvest_col)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---------- Classification / Prediction ----------
elif page == "Classification & Prediction":
    numeric_cols = get_numeric_columns(df)
    suggested_targets = [
        c for c in numeric_cols
        if any(term in c.lower() for term in ("harvest", "production", "yield", "supply", "quantity", "volume"))
    ]
    if not suggested_targets:
        st.warning("No likely numeric production target was detected. Ensure your CSV has a numeric harvest, production, yield, supply, or quantity column.")
    else:
        target = st.selectbox("Choose the value to predict", suggested_targets)
        st.caption("Models use the uploaded dataset only. Results are estimates and do not prove that a variable causes a change in production.")
        model_data = df[df[target].notna()].copy()
        # Exclude identifiers that are likely unique record IDs.
        feature_cols = []
        for col in model_data.columns:
            if col == target:
                continue
            if model_data[col].isna().all() or model_data[col].nunique(dropna=True) <= 1:
                continue
            if ("id" == col.lower().strip() or col.lower().strip().endswith("_id")) and model_data[col].nunique() > 0.8 * len(model_data):
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
                        ("model", RandomForestRegressor(
                            n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1
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
                        {"Model": "Random Forest", "MAE": mae, "RMSE": rmse, "R²": r2},
                        {
                            "Model": "Mean baseline",
                            "MAE": mean_absolute_error(y_test, baseline_prediction),
                            "RMSE": mean_squared_error(y_test, baseline_prediction) ** 0.5,
                            "R²": r2_score(y_test, baseline_prediction) if len(y_test) > 1 else np.nan,
                        }
                    ])
                    st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)

                    st.markdown("### Predictor importance")
                    with st.spinner("Calculating permutation importance..."):
                        importance_result = permutation_importance(
                            reg, X_test, y_test, n_repeats=5, random_state=42,
                            scoring="neg_mean_absolute_error", n_jobs=-1
                        )
                    importance = pd.DataFrame({
                        "Predictor": X_test.columns,
                        "Importance": importance_result.importances_mean
                    }).sort_values("Importance", ascending=False)
                    st.dataframe(importance, use_container_width=True, hide_index=True)
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

                st.markdown("### Classification: low, medium, or high")
                try:
                    class_data = model_data.copy()
                    class_y = pd.to_numeric(class_data[target], errors="coerce")
                    class_data = class_data.loc[class_y.notna()].copy()
                    class_y = class_y.loc[class_y.notna()]
                    class_labels = pd.qcut(class_y, q=3, labels=["Low", "Medium", "High"], duplicates="drop")
                    class_y = pd.Series(class_labels, index=class_y.index).astype("string")
                    class_X = class_data.loc[class_y.index, feature_cols].drop(columns=[target], errors="ignore")
                    if class_y.nunique() < 2 or class_y.value_counts().min() < 2:
                        st.info("Classification skipped because the target does not contain enough examples in multiple harvest levels.")
                    else:
                        stratify_y = class_y if class_y.value_counts().min() >= 2 else None
                        Xtr, Xte, ytr, yte = train_test_split(
                            class_X, class_y, test_size=test_fraction, random_state=42, stratify=stratify_y
                        )
                        clf = Pipeline([
                            ("preprocess", make_preprocessor(Xtr)),
                            ("model", RandomForestClassifier(
                                n_estimators=200, random_state=42,
                                class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1
                            )),
                        ])
                        clf.fit(Xtr, ytr)
                        class_prediction = clf.predict(Xte)
                        ca, cb = st.columns(2)
                        ca.metric("Accuracy", f"{accuracy_score(yte, class_prediction):.1%}")
                        cb.metric("Balanced accuracy", f"{balanced_accuracy_score(yte, class_prediction):.1%}")
                        report = pd.DataFrame(classification_report(
                            yte, class_prediction, output_dict=True, zero_division=0
                        )).T
                        st.dataframe(report.round(3), use_container_width=True)
                        labels = sorted(class_y.dropna().unique())
                        cm = confusion_matrix(yte, class_prediction, labels=labels)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.imshow(cm)
                        ax.set_xticks(range(len(labels)), labels, rotation=30)
                        ax.set_yticks(range(len(labels)), labels)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion matrix")
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as exc:
                    st.info(f"Classification was skipped for this dataset: {exc}")

# ---------- Forecasting ----------
elif page == "Harvest Forecasting":
    st.markdown("### Monthly harvest forecast")
    year_col = next((c for c in df.columns if c.lower().strip() in ("year", "harvest year", "production year")), None)
    month_col = next((c for c in df.columns if c.lower().strip() in ("month", "harvest month", "production month")), None)
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
                st.dataframe(future.round(2), use_container_width=True, hide_index=True)

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
        use_container_width=True,
    )

    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Missing (%)": [(df[c].isna().mean() * 100).round(2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.download_button(
        "Download data quality report (CSV)",
        data=quality.to_csv(index=False).encode("utf-8-sig"),
        file_name="data_quality_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    forecast = st.session_state.get("forecast_download")
    if forecast is not None:
        st.download_button(
            "Download generated forecast (CSV)",
            data=forecast.to_csv(index=False).encode("utf-8-sig"),
            file_name="coconut_production_forecast.csv",
            mime="text/csv",
            use_container_width=True,
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
            use_container_width=True,
        )

st.markdown("---")
st.caption("Coconut Production Analytics • Uploads are processed by this app for the current session. Model results depend on the quality, quantity, and coverage of the uploaded records.")
'''
out = Path("/mnt/data/app.py")
out.write_text(code, encoding="utf-8")

requirements = """streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
"""
req = Path("/mnt/data/requirements.txt")
req.write_text(requirements, encoding="utf-8")

# Compile for syntax verification.
compile(code, str(out), "exec")
print(f"Created {out} ({len(code.splitlines())} lines)")
print(f"Created {req}")
print("Python syntax check passed.")
