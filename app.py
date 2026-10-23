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
