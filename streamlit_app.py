import os
import json
import requests
import pandas as pd
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="Student Grade Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom modern CSS styling to match the premium web design
st.markdown("""
<style>
    /* Global aesthetics */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(129, 140, 248, 0.4);
        color: #c7d2fe;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(to right, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        max-width: 800px;
    }

    /* Column Cards */
    .dashboard-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 10px 20px -3px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Grade Badges */
    .grade-display {
        text-align: center;
        padding: 28px 20px;
        border-radius: 14px;
        background: #1e293b;
        border: 2px dashed #334155;
        margin-bottom: 20px;
    }
    .grade-badge {
        display: inline-block;
        font-size: 3.8rem;
        font-weight: 900;
        line-height: 1;
        padding: 14px 36px;
        border-radius: 18px;
        margin-bottom: 10px;
    }
    .grade-high {
        background: #064e3b;
        color: #34d399;
        border: 2px solid #059669;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.25);
    }
    .grade-medium {
        background: #1e3a8a;
        color: #60a5fa;
        border: 2px solid #2563eb;
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.25);
    }
    .grade-low {
        background: #78350f;
        color: #fbbf24;
        border: 2px solid #d97706;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.25);
    }
    .grade-fail {
        background: #7f1d1d;
        color: #f87171;
        border: 2px solid #dc2626;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
    }
    .grade-status {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
    }

    /* Probabilities bar styling */
    .prob-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# Default backend URL
API_URL = "http://127.0.0.1:8000/predict"

# Header Banner
st.markdown("""
<div class="header-card">
    <div class="header-badge">⚡ Scikit-Learn Model • FastAPI Backend</div>
    <div class="header-title">Student Grade Prediction Dashboard</div>
    <div class="header-subtitle">
        Predict academic letter grades and performance standings in real time using classroom engagement, demographic attributes, and study habits.
    </div>
</div>
""", unsafe_allow_html=True)

# Define presets
PRESETS = {
    "high": {
        "student_id": "STUDENT_TOP",
        "student_age": "18",
        "sex": "Female",
        "high_school_type": "Private",
        "scholarship": "100%",
        "transportation": "Private",
        "weekly_study_hours": 12.0,
        "attendance": "Always",
        "notes": "Yes",
        "listening_in_class": "Yes",
        "reading": "Yes",
        "project_work": "Yes",
        "additional_work": "No",
        "sports_activity": "Yes"
    },
    "average": {
        "student_id": "STUDENT_AVG",
        "student_age": "19-22",
        "sex": "Male",
        "high_school_type": "State",
        "scholarship": "50%",
        "transportation": "Bus",
        "weekly_study_hours": 2.0,
        "attendance": "Always",
        "notes": "Yes",
        "listening_in_class": "Yes",
        "reading": "No",
        "project_work": "No",
        "additional_work": "No",
        "sports_activity": "No"
    },
    "at_risk": {
        "student_id": "STUDENT_RISK",
        "student_age": "23-27",
        "sex": "Male",
        "high_school_type": "Other",
        "scholarship": "25%",
        "transportation": "Bus",
        "weekly_study_hours": 0.0,
        "attendance": "Never",
        "notes": "No",
        "listening_in_class": "No",
        "reading": "No",
        "project_work": "No",
        "additional_work": "Yes",
        "sports_activity": "No"
    }
}

# Initialize session state with default values if not already present
for key, val in PRESETS["average"].items():
    if key not in st.session_state:
        st.session_state[key] = val

def apply_preset(preset_key: str):
    for k, v in PRESETS[preset_key].items():
        st.session_state[k] = v

# Two-column dashboard layout
col_left, col_right = st.columns([1.25, 1.0], gap="large")

# ==========================================
# LEFT COLUMN: Input Form & Presets
# ==========================================
with col_left:
    st.markdown('<div class="card-title">📝 Student Information Input</div>', unsafe_allow_html=True)
    
    st.caption("Quickly populate the form using one of these student presets:")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("🌟 High Achiever", use_container_width=True):
            apply_preset("high")
            st.rerun()
    with btn_col2:
        if st.button("⚖️ Average Student", use_container_width=True):
            apply_preset("average")
            st.rerun()
    with btn_col3:
        if st.button("⚠️ At-Risk Student", use_container_width=True):
            apply_preset("at_risk")
            st.rerun()

    st.write("---")

    with st.form("prediction_form", clear_on_submit=False):
        # Section 1: Demographics
        st.markdown("##### 👤 Demographics & Background")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            student_id = st.text_input("Student ID", value=st.session_state.get("student_id", "STUDENT101"))
            student_age = st.selectbox(
                "Age Group",
                options=["18", "19-22", "23-27"],
                index=["18", "19-22", "23-27"].index(st.session_state.get("student_age", "19-22"))
            )
            sex = st.selectbox(
                "Gender / Sex",
                options=["Male", "Female"],
                index=["Male", "Female"].index(st.session_state.get("sex", "Male"))
            )
        with d_col2:
            high_school_type = st.selectbox(
                "High School Type",
                options=["State", "Private", "Other"],
                index=["State", "Private", "Other"].index(st.session_state.get("high_school_type", "State"))
            )
            scholarship = st.selectbox(
                "Scholarship Level",
                options=["25%", "50%", "75%", "100%"],
                index=["25%", "50%", "75%", "100%"].index(st.session_state.get("scholarship", "50%"))
            )
            transportation = st.selectbox(
                "Transportation",
                options=["Private", "Bus"],
                index=["Private", "Bus"].index(st.session_state.get("transportation", "Private"))
            )

        st.markdown("##### 📚 Study Habits & Engagement")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            weekly_study_hours = st.number_input(
                "Weekly Study Hours",
                min_value=0.0,
                max_value=60.0,
                value=float(st.session_state.get("weekly_study_hours", 2.0)),
                step=1.0
            )
            attendance = st.selectbox(
                "Attendance Frequency",
                options=["Always", "Sometimes", "Never"],
                index=["Always", "Sometimes", "Never"].index(st.session_state.get("attendance", "Always"))
            )
            notes = st.selectbox(
                "Takes Notes in Class",
                options=["Yes", "No"],
                index=["Yes", "No"].index(st.session_state.get("notes", "Yes"))
            )
        with s_col2:
            listening_in_class = st.selectbox(
                "Listens Attentively in Class",
                options=["Yes", "No"],
                index=["Yes", "No"].index(st.session_state.get("listening_in_class", "Yes"))
            )
            reading = st.selectbox(
                "Regular Reading Habits",
                options=["Yes", "No"],
                index=["Yes", "No"].index(st.session_state.get("reading", "No"))
            )
            project_work = st.selectbox(
                "Project Work Participation",
                options=["Yes", "No"],
                index=["Yes", "No"].index(st.session_state.get("project_work", "No"))
            )

        st.markdown("##### 💼 Activities & Commitments")
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            additional_work = st.selectbox(
                "Has Additional Work / Job",
                options=["No", "Yes"],
                index=["No", "Yes"].index(st.session_state.get("additional_work", "No"))
            )
        with a_col2:
            sports_activity = st.selectbox(
                "Sports & Athletics",
                options=["No", "Yes"],
                index=["No", "Yes"].index(st.session_state.get("sports_activity", "No"))
            )

        submit_btn = st.form_submit_button("🔮 Predict Grade", use_container_width=True, type="primary")

# ==========================================
# Helper Function: Grade Styling
# ==========================================
def get_grade_meta(grade_str: str):
    g = grade_str.upper()
    if g in ["AA", "BA", "A"]:
        return "grade-high", "🌟 High Academic Standing / Distinction"
    elif g in ["BB", "CB", "B"]:
        return "grade-medium", "📘 Good Academic Standing"
    elif g in ["CC", "DC", "DD", "C", "D"]:
        return "grade-low", "⚠️ Satisfactory / Needs Attention"
    else:
        return "grade-fail", "🚨 Academic Risk / Probable Fail"

# ==========================================
# RIGHT COLUMN: Result Card & Analytics
# ==========================================
with col_right:
    st.markdown('<div class="card-title">🎯 Prediction & Performance Analytics</div>', unsafe_allow_html=True)
    
    if submit_btn:
        payload = {
            "Student_ID": student_id.strip() or "STUDENT",
            "Student_Age": student_age,
            "Sex": sex,
            "High_School_Type": high_school_type,
            "Scholarship": scholarship,
            "Transportation": transportation,
            "Weekly_Study_Hours": weekly_study_hours,
            "Attendance": attendance,
            "Notes": notes,
            "Listening_in_Class": listening_in_class,
            "Reading": reading,
            "Project_work": project_work,
            "Additional_Work": additional_work,
            "Sports_activity": sports_activity
        }

        # Send request to FastAPI backend with graceful fallback
        prediction_result = None
        error_msg = None

        try:
            resp = requests.post(API_URL, json=payload, timeout=4)
            if resp.status_code == 200:
                prediction_result = resp.json()
            else:
                error_msg = f"FastAPI returned error {resp.status_code}: {resp.text}"
        except Exception:
            # Fallback to local model execution if FastAPI server is not actively running
            try:
                import joblib
                model_path = os.path.join(os.path.dirname(__file__), "student_classifier_api.pkl")
                if not os.path.exists(model_path):
                    model_path = os.path.join(os.path.dirname(__file__), "student_grade_classifier.pkl")
                
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    input_df = pd.DataFrame([payload])
                    pred = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0]
                    classes = model.classes_
                    prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}
                    confidence = round(float(max(proba)) * 100, 2)
                    
                    prediction_result = {
                        "status": "success",
                        "student_id": payload["Student_ID"],
                        "predicted_grade": str(pred),
                        "confidence": confidence,
                        "confidence_percentage": f"{confidence:.1f}%",
                        "class_probabilities": prob_dict,
                        "fallback_used": True
                    }
                else:
                    error_msg = "Could not reach FastAPI at http://127.0.0.1:8000/predict and no local .pkl model found."
            except Exception as e:
                error_msg = f"Inference failed: {str(e)}"

        if error_msg:
            st.error(error_msg)
            st.info("Tip: Start the FastAPI backend with: `uvicorn main:app --reload`")
        elif prediction_result:
            pred_grade = prediction_result.get("predicted_grade", "N/A")
            confidence = float(prediction_result.get("confidence", 0.0))
            conf_str = prediction_result.get("confidence_percentage", f"{confidence:.1f}%")
            probabilities = prediction_result.get("class_probabilities", {})

            badge_class, standing_text = get_grade_meta(pred_grade)

            # Display stylish result card
            st.markdown(f"""
            <div class="grade-display">
                <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px;">
                    Predicted Letter Grade
                </div>
                <div class="grade-badge {badge_class}">
                    {pred_grade}
                </div>
                <div class="grade-status">{standing_text}</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                    Student: <strong>{payload['Student_ID']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence Metric & Progress Bar
            st.markdown("##### 📈 Model Confidence")
            m_col1, m_col2 = st.columns([1, 2])
            with m_col1:
                st.metric("Confidence Score", conf_str)
            with m_col2:
                st.write("")
                st.progress(min(1.0, max(0.0, confidence / 100.0)))

            # Class Probability Distribution Chart
            st.markdown("##### 📊 Class Probability Distribution")
            if probabilities:
                df_prob = pd.DataFrame(
                    list(probabilities.items()),
                    columns=["Grade", "Probability"]
                ).sort_values(by="Probability", ascending=False)
                
                # Percentage representation
                df_prob["Percentage"] = (df_prob["Probability"] * 100).round(1)

                # Streamlit Bar Chart
                chart_data = df_prob.set_index("Grade")["Percentage"]
                st.bar_chart(chart_data, color="#6366f1", height=230)

            # Show details expander
            with st.expander("🔍 View Raw API Response"):
                st.json(prediction_result)

    else:
        # Default placeholder when no prediction has been triggered yet
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; border-radius: 14px; background: #111827; border: 1px dashed #374151;">
            <div style="font-size: 3.5rem; margin-bottom: 12px; opacity: 0.7;">📊</div>
            <h3 style="color: #f3f4f6; margin-bottom: 8px;">Awaiting Input</h3>
            <p style="color: #94a3b8; font-size: 0.95rem; max-width: 400px; margin: 0 auto;">
                Fill out the student characteristics on the left (or pick a sample preset), then click 
                <strong style="color: #818cf8;">Predict Grade</strong> to see real-time classification results.
            </p>
        </div>
        """, unsafe_allow_html=True)
