COCONUT ANALYTICS DASHBOARD - FIXED FILES

Files:
- app.py: corrected Streamlit application. Keep this exact filename.
- requirements.txt: Python dependencies used by the app.

DEPLOY TO STREAMLIT COMMUNITY CLOUD
1. Download and extract this package.
2. In your GitHub repository coconut_analysis, replace the root app.py with this app.py.
3. Replace the root requirements.txt with this requirements.txt.
4. Commit and push the two files to the same branch configured in Streamlit Cloud (usually main).
5. In Streamlit Cloud, open the app settings and verify Main file path is exactly app.py (not a subfolder).
6. Reboot the app and open Classification & Prediction.

Fixes included:
- Classification split checks whether stratification is feasible for the selected test proportion.
- Numeric classification bins handle tied values more safely.
- Parallel model jobs limited to one worker to reduce cloud resource pressure.
- Existing Coconut Supply & Inflation page and other dashboard pages retained.
- Python syntax validation passed.

Important: I cannot push changes to your GitHub or reboot your Streamlit Cloud app from this chat. These files are ready for you to replace the repository files. If the deployed page still errors, send the red error traceback shown on the page.
