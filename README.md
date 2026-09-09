# 🛡️ Economic Crash Early-Warning System (1970 - Present)

Continuous macroeconomic stress monitor tracking 32 financial variables, yield curve maturities, IPO euphoria, national debt levels, and empirical historical crash calibration.

---

## 🌐 Free Public Web Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended — 100% Free, 2 Minutes)

1. **Push your project code to GitHub**:
   - Create a new public repository on [GitHub](https://github.com/new) named `economic-crash-monitor`.
   - Run these commands in your project folder:
     ```bash
     git init
     git add .
     git commit -m "Deploy Economic Crash Monitor"
     git branch -M main
     git remote add origin https://github.com/YOUR_GITHUB_USERNAME/economic-crash-monitor.git
     git push -u origin main
     ```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **"New app"**.
   - Select your repository (`economic-crash-monitor`), branch (`main`), and set Main file path to:
     ```
     app/dashboard/main.py
     ```
   - Click **"Deploy!"**.

3. **🎉 Live Public URL**:
   Streamlit will give you a public URL (e.g. `https://economic-crash-monitor.streamlit.app`) accessible from any phone, tablet, or desktop in the world!

---

### Option 2: Render.com / Hugging Face / Docker Container (Free)

1. Sign up at [render.com](https://render.com/).
2. Create a **New Web Service** and connect your GitHub repository.
3. Choose **Docker** as the runtime environment.
4. Render will build the included [`Dockerfile`](file:///C:/Users/tomba/.gemini/antigravity/scratch/economic-crash-monitor/Dockerfile) and launch your public web server automatically.

---

## 🚀 Running Locally

```bash
# Activate virtual environment
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Run live dashboard
.venv\Scripts\streamlit.exe run app/dashboard/main.py
```
