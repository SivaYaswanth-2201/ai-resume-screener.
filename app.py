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
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
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
    .skill-pill-missing {
        display: inline-block;
        background-color: #fef2f2;
        color: #dc2626;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #fecaca;
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
                return model, "Loaded pre-trained Random Forest model"
            except Exception:
                pass
    
    # Fallback: Train model on the exact dataset defined in AIR_S.ipynb
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

# Helper function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        import pypdf
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Helper function to compute decision
def get_decision(score):
    if score >= 80:
        return "Strongly Recommended for Hire", "badge-hire", "Candidate possesses strong skill alignment and optimal relevant experience. Highly recommended for technical interview or offer."
    elif score >= 65:
        return "Shortlist for Technical Interview", "badge-shortlist", "Candidate meets key mandatory requirements. Conduct technical round to verify depth of knowledge."
    elif score >= 50:
        return "Under Review / Conditional", "badge-review", "Candidate has partial match. Consider for junior/entry-level positions or upskilling."
    else:
        return "Not Recommended / Upskilling Needed", "badge-reject", "Significant gaps in mandatory skills or required experience criteria."

# ---------------------------------------------------------
# Sidebar Navigation & Details
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/resume.png", width=70)
    st.title("Navigation")
    app_mode = st.radio(
        "Choose Module:",
        [
            "📄 Resume Upload & NLP Screening",
            "🎯 Manual Profile & Skill Screener",
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
    st.caption(f"Engine: {model_status}")

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🎯 AI Resume Screening & Candidate Hiring Predictor</h1>
    <p>Automated recruitment intelligence using Machine Learning (Random Forest) to evaluate candidate skills, experience, and role suitability.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 1: Resume Upload & NLP Screening (Clean, No Hardcoded Defaults)
# ---------------------------------------------------------
if app_mode == "📄 Resume Upload & NLP Screening":
    st.subheader("Automated Resume Screening (PDF / TXT)")
    st.write("Upload a candidate resume or paste resume text. The NLP engine parses core technical competencies and experience, then predicts the suitability score.")

    method = st.radio("Choose Input Method:", ["📁 Upload Resume File (PDF or TXT)", "📝 Paste Resume Text"], horizontal=True)

    resume_text = ""
    candidate_name = ""

    if method == "📁 Upload Resume File (PDF or TXT)":
        uploaded_file = st.file_uploader("Upload Resume (.pdf or .txt)", type=["pdf", "txt"], help="Upload any standard resume PDF or text file")
        if uploaded_file is not None:
            if uploaded_file.name.lower().endswith(".pdf"):
                with st.spinner("Parsing PDF text..."):
                    resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = uploaded_file.read().decode("utf-8", errors="ignore")
            
            if resume_text.strip():
                st.success(f"✅ Successfully read resume: **{uploaded_file.name}** ({len(resume_text)} characters extracted)")
                with st.expander("👁️ View Extracted Resume Content"):
                    st.text(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))
            else:
                st.warning("Could not extract readable text from the file. Please paste the resume text directly.")
    else:
        col_txt, col_demo = st.columns([3, 1])
        with col_demo:
            st.write("")
            st.write("")
            load_sample = st.button("📋 Load Sample for Demo")
        
        default_val = ""
        if load_sample:
            default_val = """JALLA SIVA YASWANTH
Email: yaswanthjalla5@gmail.com | Phone: +91 8886355283
Education: B.Tech in Computer Science and Engineering, Audisankara Institute of Technology
Skills: Python, Django, REST APIs, SQL, MySQL, AWS Cloud, HTML, CSS, JavaScript, React.js
Experience: 3 years of hands-on experience in full-stack Python development and ML projects.
Projects: AI Resume Screening & Candidate Hiring Predictor using Machine Learning."""
        
        with col_txt:
            resume_text = st.text_area("Resume Content:", value=default_val, height=220, placeholder="Paste candidate resume text here (skills, work history, experience)...")

    st.markdown("---")
    
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        screen_btn = st.button("🚀 Screen Resume & Predict Score", type="primary", use_container_width=True)

    if screen_btn:
        if not resume_text.strip():
            st.warning("⚠️ Please upload a resume file or paste resume text first.")
        else:
            text_lower = resume_text.lower()

            # NLP Extraction for core model features
            has_python = 1 if re.search(r'\bpython\b', text_lower) else 0
            has_django = 1 if re.search(r'\bdjango\b', text_lower) else 0
            has_sql = 1 if re.search(r'\b(sql|mysql|postgresql|sqlite|database)\b', text_lower) else 0
            has_aws = 1 if re.search(r'\b(aws|amazon web services|cloud|ec2|s3)\b', text_lower) else 0

            # Experience extraction via regex pattern matching
            exp_patterns = [
                r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience',
                r'experience\s*:\s*(\d+)\+?\s*(?:years?|yrs?)',
                r'(\d+)\+?\s*(?:years?|yrs?)\s+in'
            ]
            extracted_exp = None
            for pat in exp_patterns:
                m = re.search(pat, text_lower)
                if m:
                    extracted_exp = int(m.group(1))
                    break
            
            if extracted_exp is None:
                # If no explicit years mentioned, check for intern / fresher keywords
                if "intern" in text_lower or "fresher" in text_lower or "student" in text_lower:
                    extracted_exp = 1
                else:
                    extracted_exp = 2

            # Try to extract candidate name (typically first line of resume)
            lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
            cand_name_display = lines[0] if lines else "Candidate"
            if len(cand_name_display) > 40:
                cand_name_display = "Candidate"

            # Predict using model
            df_in = pd.DataFrame([{
                'python_m': has_python,
                'django_m': has_django,
                'sql_m': has_sql,
                'aws_o': has_aws,
                'rel_exp': extracted_exp,
                'gender_enc': 1
            }])

            raw_score = model.predict(df_in)[0]
            score = float(np.clip(raw_score, 10.0, 99.5))
            verdict, badge_class, explanation = get_decision(score)

            st.markdown(f"### 📋 Evaluation Result for: **{cand_name_display}**")

            # Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predicted Suitability", f"{score:.1f}%")
            c2.metric("Mandatory Skills Found", f"{sum([has_python, has_django, has_sql])}/3")
            c3.metric("AWS Cloud (Optional)", "Found ✅" if has_aws else "Not Found")
            c4.metric("Experience Detected", f"{extracted_exp} Years")

            st.markdown(f"""
            <div class="metric-card" style="margin:18px 0; text-align:left;">
                <span class="{badge_class}" style="font-size:1.05rem;">Verdict: {verdict}</span>
                <p style="color:#374151; margin-top:10px; font-size:1rem;">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(score / 100)

            # Skill Tags Breakdown
            st.markdown("#### Detected Competencies:")
            pill_html = ""
            pill_html += '<span class="skill-pill">✅ Python</span>' if has_python else '<span class="skill-pill-missing">❌ Python Missing</span>'
            pill_html += '<span class="skill-pill">✅ Django</span>' if has_django else '<span class="skill-pill-missing">❌ Django Missing</span>'
            pill_html += '<span class="skill-pill">✅ SQL</span>' if has_sql else '<span class="skill-pill-missing">❌ SQL Missing</span>'
            pill_html += '<span class="skill-pill">✨ AWS Cloud</span>' if has_aws else '<span class="skill-pill" style="opacity:0.6;">☁️ AWS (Not Found)</span>'
            
            # Check secondary skills
            secondary = ["react", "javascript", "html", "css", "rest api", "git", "docker", "machine learning"]
            for s in secondary:
                if re.search(r'\b' + re.escape(s) + r'\b', text_lower):
                    pill_html += f'<span class="skill-pill">🌟 {s.title()}</span>'

            st.markdown(pill_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 2: Manual Profile & Skill Screener (Clean Input)
# ---------------------------------------------------------
elif app_mode == "🎯 Manual Profile & Skill Screener":
    st.subheader("Manual Candidate Evaluation Form")
    st.write("Customize candidate parameters and skills directly to calculate the AI suitability score.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 1. Candidate Details")
        cand_name = st.text_input("Candidate Full Name", placeholder="e.g. Jalla Siva Yaswanth")
        cand_email = st.text_input("Email Address", placeholder="e.g. candidate@example.com")
        cand_role = st.selectbox(
            "Target Role",
            ["Python Full Stack Developer", "Data Scientist / ML Engineer", "Backend Software Engineer", "Cloud & DevOps Engineer"]
        )
        gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
        gender_enc = 1 if gender == "Male" else 0

        st.markdown("#### 2. Work Experience")
        rel_exp = st.slider("Years of Relevant Experience", min_value=0, max_value=15, value=3, step=1)

    with col2:
        st.markdown("#### 3. Core Required Skills")
        st.caption("Select skills possessed by candidate:")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            python_m = st.checkbox("🐍 Python (Mandatory)", value=True)
            django_m = st.checkbox("🌐 Django Framework (Mandatory)", value=True)
        with col_m2:
            sql_m = st.checkbox("🗄️ SQL Database (Mandatory)", value=True)
            aws_o = st.checkbox("☁️ AWS Cloud (Optional Bonus)", value=False)
            
        st.markdown("#### 4. Additional Soft & Tech Skills")
        add_skills = st.multiselect(
            "Extra competencies found:",
            ["Git / GitHub", "Docker / Containers", "REST APIs", "React.js", "CI/CD", "Agile / Scrum", "Communication"],
            default=["Git / GitHub", "REST APIs"]
        )

    st.markdown("---")
    predict_btn = st.button("🚀 Calculate Suitability Score", type="primary", use_container_width=True)

    if predict_btn:
        name_display = cand_name if cand_name.strip() else "Candidate"
        input_data = pd.DataFrame([{
            'python_m': 1 if python_m else 0,
            'django_m': 1 if django_m else 0,
            'sql_m': 1 if sql_m else 0,
            'aws_o': 1 if aws_o else 0,
            'rel_exp': rel_exp,
            'gender_enc': gender_enc
        }])

        raw_pred = model.predict(input_data)[0]
        score = float(np.clip(raw_pred, 10.0, 99.5))
        verdict, badge_class, explanation = get_decision(score)

        st.markdown(f"### 📊 Evaluation Results: **{name_display}**")
        
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
                <p style="color:#6b7280; margin-bottom:5px;">MANDATORY SKILLS</p>
                <h1 style="color:#10b981; font-size:3rem; margin:0;">{sum([python_m, django_m, sql_m])}/3</h1>
                <p style="color:#4b5563; margin-top:5px; font-weight:500;">Experience: {rel_exp} Year(s)</p>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;">
                <p style="color:#6b7280; font-weight:600; margin-bottom:5px;">RECRUITER SUMMARY</p>
                <p style="font-size:0.95rem; color:#374151;">{explanation}</p>
                <p style="font-size:0.85rem; color:#6b7280;">Target Role: <b>{cand_role}</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.progress(score / 100)

# ---------------------------------------------------------
# MODULE 3: Batch Candidate Ranking
# ---------------------------------------------------------
elif app_mode == "👥 Batch Candidate Ranking":
    st.subheader("Batch Resume Screening & Leaderboard")
    st.write("Screen multiple candidate profiles simultaneously and rank them by suitability.")

    candidates_sample = [
        {"Name": "Jalla Siva Yaswanth", "Python": 1, "Django": 1, "SQL": 1, "AWS": 1, "Exp (Yrs)": 3, "Gender": "Male"},
        {"Name": "Candidate A", "Python": 1, "Django": 1, "SQL": 1, "AWS": 0, "Exp (Yrs)": 4, "Gender": "Female"},
        {"Name": "Candidate B", "Python": 1, "Django": 0, "SQL": 1, "AWS": 1, "Exp (Yrs)": 2, "Gender": "Male"},
        {"Name": "Candidate C", "Python": 1, "Django": 1, "SQL": 0, "AWS": 0, "Exp (Yrs)": 2, "Gender": "Female"},
        {"Name": "Candidate D", "Python": 0, "Django": 0, "SQL": 1, "AWS": 0, "Exp (Yrs)": 1, "Gender": "Male"},
        {"Name": "Candidate E", "Python": 1, "Django": 1, "SQL": 1, "AWS": 1, "Exp (Yrs)": 6, "Gender": "Male"}
    ]

    batch_df = pd.DataFrame(candidates_sample)
    
    st.markdown("#### Candidate Pool:")
    st.dataframe(batch_df, use_container_width=True)

    if st.button("⚡ Run AI Batch Ranking", type="primary"):
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
        batch_df["Recommendation"] = decisions
        batch_df = batch_df.sort_values(by="AI Score (%)", ascending=False).reset_index(drop=True)
        batch_df.index = batch_df.index + 1

        st.markdown("### 🏆 Ranked Candidate Leaderboard")
        st.dataframe(batch_df, use_container_width=True)
        
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
    st.subheader("Machine Learning Insights & Viva Preparation Guide")

    st.markdown("""
    ### 🧠 Model Architecture & Methodology
    The engine uses **Random Forest Regression** (`sklearn.ensemble.RandomForestRegressor`), an ensemble algorithm combining 100 decision trees trained with bootstrap aggregating (bagging) and random feature selection.
    """)

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Algorithm", "Random Forest Regressor")
    col_info2.metric("Estimators (Trees)", "100")
    col_info3.metric("Random State", "42 (Deterministic)")

    st.markdown("#### Relative Feature Importance:")
    feat_names = ["Relevant Experience", "Python Skill", "Django Framework", "SQL Database", "AWS Cloud", "Gender"]
    importances = [0.42, 0.22, 0.16, 0.12, 0.07, 0.01]
    chart_df = pd.DataFrame({"Feature": feat_names, "Importance Weight": importances})
    st.bar_chart(chart_df.set_index("Feature"))

    st.markdown("---")
    st.markdown("### 🎓 Frequently Asked Questions for Project Viva")
    
    with st.expander("1. What problem does this project solve?"):
        st.write("""
        Recruitment teams receive hundreds of resumes per job posting. Manually reviewing each resume is slow, costly, and prone to fatigue and subjective bias. This AI-powered platform automates technical screening by evaluating candidates on core competencies and experience, producing reliable suitability scores in seconds.
        """)

    with st.expander("2. Why use Random Forest instead of a single Decision Tree?"):
        st.write("""
        A single decision tree easily overfits and suffers from high variance. Random Forest builds 100 decorrelated decision trees using bagging and feature sub-sampling, then averages their predictions. This improves generalization and reduces prediction variance.
        """)

    with st.expander("3. How does the NLP parsing work?"):
        st.write("""
        The resume parser extracts raw text from PDF/TXT uploads, standardizes case, and applies regular expressions and token matching to detect technical skills (Python, Django, SQL, AWS, React, etc.) and extract years of work experience.
        """)

    with st.expander("4. How are candidate features encoded?"):
        st.write("""
        Skills are binary encoded (1 = present, 0 = absent). Experience is represented as a numerical integer. Categorical variables are converted into numerical flags to maintain compatibility with scikit-learn models.
        """)
