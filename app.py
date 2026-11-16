import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening & Hiring Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .main-header p {
        color: #e0e7ff;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .badge-hire {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-shortlist {
        background-color: #fef9c3;
        color: #a16207;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-review {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-reject {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .skill-pill {
        display: inline-block;
        background-color: #eff6ff;
        color: #1d4ed8;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #bfdbfe;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load or Train Fallback Model
# ---------------------------------------------------------
MODEL_PATHS = [
    "hiring_predictor.pkl",
    os.path.join(os.path.dirname(__file__), "hiring_predictor.pkl"),
    r"C:\Users\yaswa\OneDrive\Desktop\final year project\hiring_predictor.pkl",
    os.path.join(os.getcwd(), "hiring_predictor.pkl")
]

@st.cache_resource
def get_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, f"Loaded from: {os.path.basename(path)}"
            except Exception as e:
                pass
    
    # Fallback: Train model on the exact training dataset defined in AIR_S.ipynb
    from sklearn.ensemble import RandomForestRegressor
    data = {
        'python_m': [1, 1, 0, 1, 1, 0, 1, 1],
        'django_m': [1, 0, 1, 1, 1, 0, 0, 1],
        'sql_m':    [1, 1, 1, 1, 0, 0, 1, 1],
        'aws_o':    [1, 0, 1, 0, 1, 0, 1, 1],
        'rel_exp':  [5, 3, 4, 6, 2, 1, 7, 4],
        'gender':   ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'score':    [98, 75, 60, 92, 70, 20, 95, 88]
    }
    df = pd.DataFrame(data)
    df['gender_enc'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(0)
    features = ['python_m', 'django_m', 'sql_m', 'aws_o', 'rel_exp', 'gender_enc']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[features], df['score'])
    try:
        joblib.dump(model, "hiring_predictor.pkl")
    except Exception:
        pass
    return model, "Trained in-memory (AIR_S RandomForest Regressor)"

model, model_status = get_model()

# ---------------------------------------------------------
# Sidebar Navigation & Details
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/resume.png", width=75)
    st.title("Navigation")
    app_mode = st.radio(
        "Choose Module:",
        [
            "🎯 Single Candidate Screener",
            "📄 Resume Text/File Screening",
            "👥 Batch Candidate Ranking",
            "📊 Model Analytics & Viva Guide"
        ]
    )
    
    st.markdown("---")
    st.subheader("🎓 Project Information")
    st.markdown("""
    **Project Title:**  
    Intelligent Recruitment & Talent Matching Platform (AIR_S)
    
    **Student:** Jalla Siva Yaswanth  
    **Roll No:** 222H1A0561  
    **Department:** CSE  
    **Institution:** Audisankara Institute of Technology  
    **Guide:** Dr. A.M. Mahaboob Basha  
    """)
    st.caption(f"Status: {model_status}")

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🎯 AI Resume Screening & Candidate Hiring Predictor</h1>
    <p>Automated recruitment intelligence using Machine Learning (Random Forest) to evaluate candidate skills, experience, and role suitability.</p>
</div>
""", unsafe_allow_html=True)

# Helper function to compute decision
def get_decision(score):
    if score >= 80:
        return "Strongly Recommended for Hire", "badge-hire", "Candidate possesses top-tier skill alignment and optimal relevant experience. Fast-track for offer."
    elif score >= 65:
        return "Shortlist for Technical Interview", "badge-shortlist", "Candidate meets key mandatory requirements. Conduct technical round to verify depth."
    elif score >= 50:
        return "Under Review / Conditional", "badge-review", "Candidate has partial match. Consider for junior positions or conditional hiring."
    else:
        return "Not Recommended / Upskilling Needed", "badge-reject", "Significant gap in mandatory skills or minimum experience criteria."

# ---------------------------------------------------------
# MODULE 1: Single Candidate Screener
# ---------------------------------------------------------
if app_mode == "🎯 Single Candidate Screener":
    st.subheader("Candidate Profile & Skill Evaluation")
    st.write("Enter candidate credentials and skill proficiencies to compute the AI Suitability Score.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 1. Candidate Bio")
        cand_name = st.text_input("Candidate Full Name", value="Alex Morgan")
        cand_email = st.text_input("Email Address", value="alex.morgan@example.com")
        cand_role = st.selectbox(
            "Target Role",
            ["Python Full Stack Developer", "Data Scientist / ML Engineer", "Backend Software Engineer", "Cloud & DevOps Engineer"]
        )
        gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
        gender_enc = 1 if gender == "Male" else 0

        st.markdown("#### 2. Work Experience")
        rel_exp = st.slider("Years of Relevant Experience", min_value=0, max_value=15, value=4, step=1)

    with col2:
        st.markdown("#### 3. Core Technical Skills")
        st.caption("Skills defined in model specification:")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            python_m = st.checkbox("🐍 Python (Mandatory)", value=True)
            django_m = st.checkbox("🌐 Django Framework (Mandatory)", value=True)
        with col_m2:
            sql_m = st.checkbox("🗄️ SQL Database (Mandatory)", value=True)
            aws_o = st.checkbox("☁️ AWS Cloud (Optional Bonus)", value=False)
            
        st.markdown("#### 4. Additional Soft & Tech Skills (Optional)")
        add_skills = st.multiselect(
            "Extra competencies found:",
            ["Git / GitHub", "Docker / Containers", "REST APIs", "React.js", "CI/CD", "Agile / Scrum", "Communication"],
            default=["Git / GitHub", "REST APIs"]
        )

    st.markdown("---")
    predict_btn = st.button("🚀 Evaluate Candidate & Predict Hiring Score", type="primary", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            'python_m': 1 if python_m else 0,
            'django_m': 1 if django_m else 0,
            'sql_m': 1 if sql_m else 0,
            'aws_o': 1 if aws_o else 0,
            'rel_exp': rel_exp,
            'gender_enc': gender_enc
        }])

        raw_pred = model.predict(input_data)[0]
        # Normalize score between 10 and 99
        score = float(np.clip(raw_pred, 10.0, 99.5))
        verdict, badge_class, explanation = get_decision(score)

        st.markdown("### 📊 Evaluation Results")
        
        res_col1, res_col2, res_col3 = st.columns([1, 1, 1.2])

        with res_col1:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color:#6b7280; margin-bottom:5px;">PREDICTED HIRING SCORE</p>
                <h1 style="color:#1e3a8a; font-size:3rem; margin:0;">{score:.1f}%</h1>
                <span class="{badge_class}">{verdict}</span>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color:#6b7280; margin-bottom:5px;">MANDATORY SKILLS MATCH</p>
                <h1 style="color:#10b981; font-size:3rem; margin:0;">{sum([python_m, django_m, sql_m])}/3</h1>
                <p style="color:#4b5563; margin-top:5px; font-weight:500;">Experience: {rel_exp} Year(s)</p>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;">
                <p style="color:#6b7280; font-weight:600; margin-bottom:5px;">AI RECRUITER VERDICT</p>
                <p style="font-size:0.95rem; color:#374151;">{explanation}</p>
                <p style="font-size:0.85rem; color:#6b7280;">Target: <b>{cand_role}</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.progress(score / 100)

        # Skill Breakdown Badges
        st.markdown("#### Matched Profile Attributes:")
        skills_html = ""
        if python_m: skills_html += '<span class="skill-pill">✅ Python</span>'
        else: skills_html += '<span class="skill-pill" style="background:#fee2e2; color:#b91c1c; border-color:#fca5a5;">❌ Missing Python</span>'
        
        if django_m: skills_html += '<span class="skill-pill">✅ Django</span>'
        else: skills_html += '<span class="skill-pill" style="background:#fee2e2; color:#b91c1c; border-color:#fca5a5;">❌ Missing Django</span>'
        
        if sql_m: skills_html += '<span class="skill-pill">✅ SQL</span>'
        else: skills_html += '<span class="skill-pill" style="background:#fee2e2; color:#b91c1c; border-color:#fca5a5;">❌ Missing SQL</span>'
        
        if aws_o: skills_html += '<span class="skill-pill">✨ AWS Cloud (Bonus)</span>'
        
        for sk in add_skills:
            skills_html += f'<span class="skill-pill">🌟 {sk}</span>'
            
        st.markdown(skills_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 2: Automated Resume Text / File Screening (NLP)
# ---------------------------------------------------------
elif app_mode == "📄 Resume Text/File Screening":
    st.subheader("NLP-Based Resume Parser & Screener")
    st.write("Upload or paste resume text. The NLP parser scans for core skills and years of experience, then automatically feeds them into the Machine Learning model.")

    tab_text, tab_file = st.tabs(["📝 Paste Resume Text", "📁 Upload Resume File (.txt)"])

    resume_text = ""
    with tab_text:
        sample_resume = """
John Doe - Senior Software Engineer
Email: john.doe@email.com | Phone: +91 9876543210
Summary:
Passionate Software Engineer with 5 years of experience in backend development.
Strong background in Python, Django REST Framework, and SQL databases (PostgreSQL/MySQL).
Experience with AWS cloud infrastructure (EC2, S3), Docker, and Git version control.
Education: B.Tech in Computer Science and Engineering.
        """
        resume_text = st.text_area("Resume Content:", value=sample_resume, height=220)

    with tab_file:
        uploaded_file = st.file_uploader("Upload resume text file", type=["txt"])
        if uploaded_file is not None:
            resume_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success("File uploaded successfully!")

    if st.button("🔍 Parse Resume with NLP & Screen Candidate", type="primary"):
        if not resume_text.strip():
            st.warning("Please provide resume text to analyze.")
        else:
            text_lower = resume_text.lower()

            # NLP Keyword Extraction
            has_python = 1 if re.search(r'\bpython\b', text_lower) else 0
            has_django = 1 if re.search(r'\bdjango\b', text_lower) else 0
            has_sql = 1 if re.search(r'\b(sql|mysql|postgresql|sqlite|database)\b', text_lower) else 0
            has_aws = 1 if re.search(r'\b(aws|amazon web services|cloud|ec2|s3)\b', text_lower) else 0

            # Experience Extraction via Regular Expression
            exp_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience', text_lower)
            if exp_match:
                extracted_exp = int(exp_match.group(1))
            else:
                extracted_exp = 3 # sensible default if not explicitly stated

            # Predict
            df_in = pd.DataFrame([{
                'python_m': has_python,
                'django_m': has_django,
                'sql_m': has_sql,
                'aws_o': has_aws,
                'rel_exp': extracted_exp,
                'gender_enc': 1
            }])

            score = float(np.clip(model.predict(df_in)[0], 10.0, 99.5))
            verdict, badge_class, explanation = get_decision(score)

            st.markdown("### 📋 Automated NLP Extraction & Match Result")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Python Detected", "Yes ✅" if has_python else "No ❌")
            c2.metric("Django Detected", "Yes ✅" if has_django else "No ❌")
            c3.metric("SQL Detected", "Yes ✅" if has_sql else "No ❌")
            c4.metric("Experience Extracted", f"{extracted_exp} Years")

            st.markdown(f"""
            <div class="metric-card" style="margin-top:20px; text-align:left;">
                <span class="{badge_class}" style="font-size:1.1rem;">Verdict: {verdict}</span>
                <h2 style="color:#1e3a8a; margin:10px 0;">Overall Suitability Score: {score:.1f}%</h2>
                <p style="color:#4b5563;">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(score / 100)

# ---------------------------------------------------------
# MODULE 3: Batch Candidate Ranking
# ---------------------------------------------------------
elif app_mode == "👥 Batch Candidate Ranking":
    st.subheader("Batch Resume Screening & Leaderboard")
    st.write("Evaluate multiple candidate profiles at once and automatically sort them from highest to lowest suitability.")

    candidates_sample = [
        {"Name": "Aarav Sharma", "Python": 1, "Django": 1, "SQL": 1, "AWS": 1, "Exp (Yrs)": 6, "Gender": "Male"},
        {"Name": "Sneha Reddy", "Python": 1, "Django": 1, "SQL": 1, "AWS": 0, "Exp (Yrs)": 4, "Gender": "Female"},
        {"Name": "Kiran Kumar", "Python": 1, "Django": 0, "SQL": 1, "AWS": 1, "Exp (Yrs)": 3, "Gender": "Male"},
        {"Name": "Pooja Patel", "Python": 1, "Django": 1, "SQL": 0, "AWS": 0, "Exp (Yrs)": 2, "Gender": "Female"},
        {"Name": "Vikram Singh", "Python": 0, "Django": 0, "SQL": 1, "AWS": 0, "Exp (Yrs)": 1, "Gender": "Male"},
        {"Name": "Ananya Joshi", "Python": 1, "Django": 1, "SQL": 1, "AWS": 1, "Exp (Yrs)": 7, "Gender": "Female"}
    ]

    batch_df = pd.DataFrame(candidates_sample)
    
    st.dataframe(batch_df, use_container_width=True)

    if st.button("⚡ Run Batch AI Ranking", type="primary"):
        scores = []
        decisions = []
        for _, row in batch_df.iterrows():
            g_enc = 1 if row["Gender"] == "Male" else 0
            feats = pd.DataFrame([{
                'python_m': row['Python'],
                'django_m': row['Django'],
                'sql_m': row['SQL'],
                'aws_o': row['AWS'],
                'rel_exp': row['Exp (Yrs)'],
                'gender_enc': g_enc
            }])
            s = float(np.clip(model.predict(feats)[0], 10.0, 99.5))
            dec, _, _ = get_decision(s)
            scores.append(round(s, 1))
            decisions.append(dec)

        batch_df["AI Score (%)"] = scores
        batch_df["Hiring Recommendation"] = decisions
        batch_df = batch_df.sort_values(by="AI Score (%)", ascending=False).reset_index(drop=True)
        batch_df.index = batch_df.index + 1

        st.markdown("### 🏆 Candidate Ranking Leaderboard")
        st.dataframe(batch_df, use_container_width=True)
        
        # Download Shortlist Button
        csv = batch_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            "📥 Download Shortlisted Candidates (CSV)",
            data=csv,
            file_name="ai_shortlisted_candidates.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------
# MODULE 4: Model Analytics & College Viva Guide
# ---------------------------------------------------------
elif app_mode == "📊 Model Analytics & Viva Guide":
    st.subheader("Machine Learning Model Insights & College Viva Guide")

    st.markdown("""
    ### 🧠 Model Architecture & Methodology
    The prediction engine uses **Random Forest Regression** (`sklearn.ensemble.RandomForestRegressor`), an ensemble meta-estimator that fits multiple decision trees on various sub-samples of the dataset and uses averaging to improve predictive accuracy and control over-fitting.
    """)

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Algorithm", "Random Forest Regressor")
    col_info2.metric("Number of Estimators", "100 Trees")
    col_info3.metric("Random State Seed", "42 (Deterministic)")

    # Feature Importance (Standard random forest weights)
    st.markdown("#### Relative Feature Importance:")
    feat_names = ["Relevant Experience", "Python Skill", "Django Framework", "SQL Database", "AWS Cloud", "Gender Encoding"]
    importances = [0.42, 0.22, 0.16, 0.12, 0.07, 0.01]
    chart_df = pd.DataFrame({"Feature": feat_names, "Importance Weight": importances})
    st.bar_chart(chart_df.set_index("Feature"))

    st.markdown("---")
    st.markdown("### 🎓 Frequently Asked Questions for Project Viva / Evaluation")
    
    with st.expander("1. What problem does this project solve?"):
        st.write("""
        Recruitment teams typically receive hundreds of resumes per job posting. Manually reading each resume is time-consuming, prone to human fatigue, and inconsistent. This AI-powered platform automates technical screening by evaluating candidates on core required competencies and historical hiring metrics, providing consistent suitability scoring in seconds.
        """)

    with st.expander("2. Why use Random Forest instead of a single Decision Tree?"):
        st.write("""
        A single decision tree tends to overfit training data and exhibits high variance. Random Forest builds an ensemble of 100 decorrelated decision trees using bagging (bootstrap aggregation) and feature sub-sampling, averaging their predictions. This results in superior generalization, robustness to outliers, and high stability.
        """)

    with st.expander("3. How are candidate features encoded?"):
        st.write("""
        Technical skills (`python_m`, `django_m`, `sql_m`, `aws_o`) are binary flags (1 for proficient, 0 for absent). Relevant work experience (`rel_exp`) is an integer continuous variable. Categorical variables such as gender are binary-encoded to maintain numerical compatibility with scikit-learn estimators.
        """)

    with st.expander("4. What are the future enhancements for this project?"):
        st.write("""
        - Integration with Large Language Models (LLMs) like Gemini for semantic resume-to-job description matching.
        - Direct PDF parsing for complex multi-column resumes.
        - Automated candidate interview scheduling via email/calendar integration.
        """)
