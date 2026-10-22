# app.py
# Coconut Production Analytics Dashboard
# Run: streamlit run app.py

from pathlib import Path
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
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.inspection import permutation_importance

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
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .main-title {font-size: 2.25rem; font-weight: 750; letter-spacing: -0.04em;}
    .subtle {color: #737783;}
    div[data-testid="stMetric"] {
        background: #f7f8fb; border: 1px solid #e7e9ef;
        padding: 14px 16px; border-radius: 12px;
    }
    section[data-testid="stSidebar"] {border-right: 1px solid #e7e9ef;}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🥥 Coconut Analytics")
st.sidebar.caption("Production • Statistics • Prediction")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV dataset", type=["csv"], help="Upload a CSV file containing coconut production records."
)

with st.sidebar.expander("CSV and cleaning settings", expanded=False):
    remove_duplicates = st.checkbox("Remove exact duplicate rows", value=True)
    normalize_text = st.checkbox("Trim text and standardize blanks", value=True)
    show_preview_rows = st.selectbox("Preview rows", [5, 10, 20, 50], index=1)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Analysis",
    [
        "1. Overview and cleaning",
        "2. Statistics and EDA",
        "3. Classification and prediction",
        "4. Forecasting",
        "5. Downloads",
    ],
)

st.markdown('<div class="main-title">Coconut Production Analysis</div>', unsafe_allow_html=True)
st.caption("Explore production records, review data quality, compare machine-learning models, and forecast monthly harvest.")

# Load uploaded data or an optional local default dataset.
raw = None
if uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file, encoding="cp1252")
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
else:
    candidates = [
        Path(__file__).resolve().parent / "Combine Data.csv",
        Path.cwd() / "Combine Data.csv",
    ]
    default_path = next((p for p in candidates if p.exists()), None)
    if default_path:
        try:
            raw = pd.read_csv(default_path, encoding="utf-8-sig")
            st.sidebar.info(f"Using local file: {default_path.name}")
        except Exception as exc:
            st.error(f"Could not read {default_path.name}: {exc}")
            st.stop()

if raw is None:
    st.info("Upload your coconut production CSV using the left sidebar to begin.")
    st.markdown("### Suggested dataset fields")
    st.code("Year, Month, Barangay, Harvested Area (ha), Harvest (kg), Rainfall (mm), Temperature (°C), Weather")
    st.stop()

if raw.empty:
    st.error("The uploaded CSV has no data rows.")
    st.stop()

# Basic cleaning
df = raw.copy()
df.columns = [str(c).strip() for c in df.columns]
if normalize_text:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({
            "": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA, "None": pd.NA
        })
duplicate_count = int(df.duplicated().sum())
if remove_duplicates:
    df = df.drop_duplicates().copy()

numeric_candidates = [
    "Year", "Harvested Area (ha)", "Harvest (kg)", "Farmgate Price (PHP/kg)",
    "Copra Price (PHP/kg)", "Whole Nut Price (PHP/kg)", "Rainfall (mm)",
    "Temperature (°C)", "Inflation Rate (%)"
]
for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

months = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
month_map = {m.lower(): i for i, m in enumerate(months, 1)}
if "Month" in df.columns:
    df["Month"] = df["Month"].astype("string").str.strip().str.title()
    if "Month Number" not in df.columns:
        df["Month Number"] = df["Month"].str.lower().map(month_map)

for col in ["Harvested Area (ha)", "Harvest (kg)", "Farmgate Price (PHP/kg)",
            "Copra Price (PHP/kg)", "Whole Nut Price (PHP/kg)", "Rainfall (mm)"]:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

st.markdown(f"**Dataset:** `{uploaded_file.name if uploaded_file else 'Combine Data.csv' if len(df) else 'Uploaded dataset'}`")
if page == "1. Overview and cleaning":
    st.markdown("### Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Exact duplicates found", f"{duplicate_count:,}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    st.markdown("### Dataset preview")
    st.dataframe(df.head(show_preview_rows), use_container_width=True)
    st.markdown("### Data quality report")
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Missing (%)": [(df[c].isna().mean() * 100).round(2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.caption("Cleaning removes exact duplicate rows only when selected. Negative production, area, price, and rainfall values are marked missing rather than silently replaced.")

elif page == "2. Statistics and EDA":
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.markdown("### Descriptive statistics")
        st.dataframe(df[numeric_cols].describe().T.round(3), use_container_width=True)
        st.markdown("### Correlation between numeric fields")
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(7, len(numeric_cols) * 0.8), 5.5))
        im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=70, ha="right")
        ax.set_yticks(range(len(corr.index)), corr.index)
        fig.colorbar(im, ax=ax, label="Correlation")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        selected_num = st.selectbox("Explore numeric distribution", numeric_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[selected_num].dropna(), bins=25, edgecolor="white")
        ax.set_title(f"Distribution of {selected_num}")
        ax.set_xlabel(selected_num)
        ax.set_ylabel("Records")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No numeric columns were detected in this dataset.")

    category_cols = [c for c in ["Barangay", "Weather", "Month", "Road Condition"] if c in df.columns]
    if category_cols:
        st.markdown("### Category breakdown")
        selected_cat = st.selectbox("Choose a category", category_cols)
        counts = df[selected_cat].fillna("Missing").value_counts().head(20)
        st.dataframe(counts.rename("Records").to_frame(), use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Records by {selected_cat}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if {"Barangay", "Harvest (kg)"}.issubset(df.columns):
        st.markdown("### Total harvest by barangay")
        by_barangay = df.groupby("Barangay", dropna=False)["Harvest (kg)"].sum(min_count=1).sort_values(ascending=False)
        st.dataframe(by_barangay.rename("Total harvest (kg)").to_frame(), use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        by_barangay.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Harvest (kg)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

elif page == "3. Classification and prediction":
    target = "Harvest (kg)"
    required = ["Year", "Month", "Barangay", target]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        st.warning("Prediction needs these columns: " + ", ".join(required) + ". Missing: " + ", ".join(missing_required))
    else:
        model_df = df[df[target].notna()].copy()
        if len(model_df) < 10:
            st.warning("At least 10 records with a valid Harvest (kg) value are recommended for a basic train/test demonstration.")
        else:
            # Classification categories are derived from harvest quantity.
            try:
                model_df["Harvest Level"] = pd.qcut(
                    model_df[target], q=3, labels=["Low", "Medium", "High"], duplicates="drop"
                )
            except ValueError:
                model_df["Harvest Level"] = pd.cut(model_df[target], bins=3, labels=["Low", "Medium", "High"])
            model_df = model_df.dropna(subset=["Harvest Level"])
            feature_cols = [c for c in model_df.columns if c not in {target, "Harvest Level"}]
            feature_cols = [c for c in feature_cols if not model_df[c].isna().all()]
            if not feature_cols or len(model_df) < 5:
                st.warning("There are not enough usable predictors or records to train a model.")
            else:
                X = model_df[feature_cols].copy()
                y_reg = model_df[target].astype(float)
                y_clf = model_df["Harvest Level"].astype(str)
                numeric_features = X.select_dtypes(include=np.number).columns.tolist()
                categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
                preprocessor = ColumnTransformer([
                    ("num", Pipeline([
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]), numeric_features),
                    ("cat", Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]), categorical_features),
                ], remainder="drop")

                st.markdown("### Harvest-level classification")
                st.caption("Low, Medium, and High classes are derived from the distribution of harvest quantity. Metrics describe prediction, not causation.")
                st.write("Class counts")
                st.dataframe(y_clf.value_counts().rename("Records").to_frame(), use_container_width=True)
                if y_clf.nunique() >= 2 and y_clf.value_counts().min() >= 2:
                    stratify = y_clf
                else:
                    stratify = None
                try:
                    Xtr, Xte, ytr, yte = train_test_split(
                        X, y_clf, test_size=0.2, random_state=42, stratify=stratify
                    )
                    clf = Pipeline([
                        ("preprocess", preprocessor),
                        ("model", RandomForestClassifier(
                            n_estimators=200, random_state=42, class_weight="balanced_subsample",
                            min_samples_leaf=2, n_jobs=-1
                        )),
                    ])
                    clf.fit(Xtr, ytr)
                    pred = clf.predict(Xte)
                    m1, m2 = st.columns(2)
                    m1.metric("Accuracy", f"{accuracy_score(yte, pred):.1%}")
                    m2.metric("Balanced accuracy", f"{balanced_accuracy_score(yte, pred):.1%}")
                    report = pd.DataFrame(classification_report(yte, pred, output_dict=True, zero_division=0)).T
                    st.dataframe(report.round(3), use_container_width=True)
                    labels = sorted(y_clf.unique())
                    cm = confusion_matrix(yte, pred, labels=labels)
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
                    st.error(f"Classification could not run on this dataset: {exc}")

                st.markdown("### Harvest quantity regression")
                try:
                    Xtr, Xte, ytr, yte = train_test_split(X, y_reg, test_size=0.2, random_state=42)
                    reg = Pipeline([
                        ("preprocess", preprocessor),
                        ("model", RandomForestRegressor(
                            n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1
                        )),
                    ])
                    reg.fit(Xtr, ytr)
                    pred_r = reg.predict(Xte)
                    mae = mean_absolute_error(yte, pred_r)
                    rmse = mean_squared_error(yte, pred_r) ** 0.5
                    r2 = r2_score(yte, pred_r) if len(yte) > 1 else float("nan")
                    a, b, c = st.columns(3)
                    a.metric("MAE (kg)", f"{mae:,.2f}")
                    b.metric("RMSE (kg)", f"{rmse:,.2f}")
                    c.metric("R²", f"{r2:.3f}" if np.isfinite(r2) else "N/A")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(yte, pred_r, alpha=0.7)
                    lo = min(float(yte.min()), float(np.min(pred_r)))
                    hi = max(float(yte.max()), float(np.max(pred_r)))
                    ax.plot([lo, hi], [lo, hi], linestyle="--")
                    ax.set_xlabel("Actual harvest (kg)")
                    ax.set_ylabel("Predicted harvest (kg)")
                    ax.set_title("Actual vs predicted harvest")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    baseline = DummyRegressor(strategy="mean")
                    baseline.fit(Xtr, ytr)
                    base_pred = baseline.predict(Xte)
                    st.markdown("**Comparison with a simple baseline**")
                    st.dataframe(pd.DataFrame([
                        {"Model": "Random Forest", "MAE": mae, "RMSE": rmse, "R²": r2},
                        {"Model": "Mean baseline", "MAE": mean_absolute_error(yte, base_pred),
                         "RMSE": mean_squared_error(yte, base_pred) ** 0.5,
                         "R²": r2_score(yte, base_pred) if len(yte) > 1 else np.nan},
                    ]).round(3), use_container_width=True, hide_index=True)
                    imp = permutation_importance(
                        reg, Xte, yte, n_repeats=5, random_state=42,
                        scoring="neg_mean_absolute_error", n_jobs=-1
                    )
                    importance = pd.DataFrame({
                        "Feature": Xte.columns,
                        "Importance": imp.importances_mean,
                    }).sort_values("Importance", ascending=False)
                    st.markdown("**Most useful predictors (permutation importance)**")
                    st.dataframe(importance, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"Regression could not run on this dataset: {exc}")

elif page == "4. Forecasting":
    st.markdown("### Monthly harvest forecast")
    needed = {"Year", "Month", "Harvest (kg)"}
    if not needed.issubset(df.columns):
        st.warning("Forecasting requires the columns: Year, Month, and Harvest (kg).")
    else:
        fc = df[["Year", "Month", "Harvest (kg)"]].copy()
        fc["Year"] = pd.to_numeric(fc["Year"], errors="coerce")
        fc["Month Number"] = fc["Month"].astype("string").str.title().str.lower().map(month_map)
        fc["Harvest (kg)"] = pd.to_numeric(fc["Harvest (kg)"], errors="coerce")
        fc = fc.dropna(subset=["Year", "Month Number", "Harvest (kg)"])
        if fc.empty:
            st.warning("No valid year/month/harvest records were found.")
        else:
            fc["Date"] = pd.to_datetime({
                "year": fc["Year"].astype(int),
                "month": fc["Month Number"].astype(int),
                "day": 1,
            })
            monthly = fc.groupby("Date")["Harvest (kg)"].sum().sort_index().asfreq("MS")
            missing_months = int(monthly.isna().sum())
            monthly = monthly.interpolate(method="time", limit_direction="both")
            st.metric("Monthly observations", len(monthly))
            st.caption(f"Calendar months with no recorded total before interpolation: {missing_months}. Forecast quality depends on record coverage.")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(monthly.index, monthly.values)
            ax.set_title("Historical monthly total harvest")
            ax.set_ylabel("Harvest (kg)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if len(monthly) < 6:
                st.warning("There are too few monthly observations for a reliable forecast. Add more historical data.")
            else:
                horizon = st.slider("Months to forecast", min_value=3, max_value=24, value=12)
                future_index = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
                forecast_values = None
                method = "Seasonal-naive / recent-year pattern"
                if HAS_STATSMODELS and len(monthly) >= 24:
                    try:
                        fit = ExponentialSmoothing(
                            monthly, trend="add", seasonal="add", seasonal_periods=12,
                            initialization_method="estimated"
                        ).fit(optimized=True)
                        forecast_values = np.asarray(fit.forecast(horizon))
                        method = "Exponential Smoothing"
                    except Exception:
                        forecast_values = None
                if forecast_values is None:
                    forecast_values = []
                    for date in future_index:
                        lag_date = date - pd.DateOffset(months=12)
                        if lag_date in monthly.index:
                            forecast_values.append(float(monthly.loc[lag_date]))
                        else:
                            forecast_values.append(float(monthly.tail(min(12, len(monthly))).median()))
                    forecast_values = np.asarray(forecast_values)
                forecast_values = np.maximum(forecast_values, 0)
                future = pd.DataFrame({
                    "Date": future_index,
                    "Forecast Harvest (kg)": forecast_values,
                    "Method": method,
                })
                st.write(f"Forecast method: **{method}**")
                st.dataframe(future.round(2), use_container_width=True, hide_index=True)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(monthly.index, monthly.values, label="Historical")
                ax.plot(future_index, forecast_values, marker="o", label="Forecast")
                ax.set_title(f"Next {horizon} months")
                ax.set_ylabel("Harvest (kg)")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.session_state["forecast_download"] = future

elif page == "5. Downloads":
    st.markdown("### Download analysis data")
    st.write("Export the cleaned dataset or the currently generated forecast.")
    st.download_button(
        "Download cleaned dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="cleaned_coconut_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
    forecast_data = st.session_state.get("forecast_download")
    if forecast_data is not None:
        st.download_button(
            "Download forecast (CSV)",
            data=forecast_data.to_csv(index=False).encode("utf-8-sig"),
            file_name="coconut_harvest_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("To create a forecast download, open Forecasting first and generate a forecast.")
    st.download_button(
        "Download data quality report (CSV)",
        data=pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(df[c].dtype) for c in df.columns],
            "Missing values": [int(df[c].isna().sum()) for c in df.columns],
            "Missing (%)": [(df[c].isna().mean() * 100).round(2) for c in df.columns],
        }).to_csv(index=False).encode("utf-8-sig"),
        file_name="data_quality_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")
st.caption("Coconut Production Analytics • Results depend on the quality and coverage of the uploaded dataset. Predictive relationships do not prove causation.")
