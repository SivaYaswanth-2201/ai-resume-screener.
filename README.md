# 🎯 AI Resume Screening & Candidate Hiring Predictor (AIR_S)

### Final Year Project
- **Student Name:** Jalla Siva Yaswanth
- **Registration / Roll No:** 222H1A0561
- **Department:** Computer Science & Engineering (CSE)
- **Institution:** Audisankara Institute of Technology (Autonomous), Gudur
- **Faculty Guide:** Dr. A.M. Mahaboob Basha (Associate Professor, CSE)
- **Batch:** 2022 – 2026

---

## 📌 Project Overview
An automated recruitment intelligence system powered by Machine Learning (Random Forest Regression) that parses candidate qualifications, extracts technical proficiencies and work experience, and computes a standardized **Hiring Suitability Score (0–100%)**.

### Core Features:
1. **Single Candidate Screener:** Interactive evaluation with sliders and checkboxes for skills (Python, Django, SQL, AWS) and experience.
2. **NLP-Based Resume Screening:** Automatically parses resume text to extract competencies and experience using Regex and NLP tokenization.
3. **Batch Screening Leaderboard:** Evaluates multiple applicants concurrently and ranks them from best to lowest fit with CSV export.
4. **Model Analytics & Viva Preparation:** Interactive charts showing feature importance weights, Random Forest architectural explanations, and frequently asked viva questions with answers.

---

## 🚀 How to Run Locally

### Option 1: Double-Click Launcher (Easiest)
Simply double-click the file named **`run_app.bat`** located in this folder. It will automatically launch the Streamlit server and open the web app in your default browser.

### Option 2: Command Line
1. Open PowerShell or Command Prompt in this folder.
2. Run:
```bash
python -m streamlit run app.py
```
3. Open your browser and navigate to:
```
http://localhost:8501
```

---

## 🌐 How to Deploy Online for Free (Live Web Link)

You can deploy this application for free on **Streamlit Community Cloud** to share a live URL with your professors, team members, and evaluators:

1. Create a free account at [GitHub](https://github.com) if you do not already have one.
2. Create a new repository (e.g., `ai-resume-screener`).
3. Upload the following 4 files:
   - `app.py`
   - `hiring_predictor.pkl`
   - `requirements.txt`
   - `README.md`
4. Visit [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
5. Click **"New App"**, select your repository, set the main file path to `app.py`, and click **Deploy**!
6. In about 2 minutes, you will receive a public URL (e.g., `https://ai-resume-screener-yaswanth.streamlit.app`) that anyone can open from their phone or laptop.

---

## 🧠 Model Specifications
- **Algorithm:** Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)
- **Trees (Estimators):** 100
- **Random State:** 42
- **Features:**
  1. `python_m` (Python Mandatory)
  2. `django_m` (Django Mandatory)
  3. `sql_m` (SQL Mandatory)
  4. `aws_o` (AWS Cloud Optional)
  5. `rel_exp` (Relevant Experience in Years)
  6. `gender_enc` (Gender Binary Encoding)
- **Target Variable:** `score` (Hiring Suitability Score: 0 to 100)
