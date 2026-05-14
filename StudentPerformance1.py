import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import google.generativeai as genai

from gtts import gTTS
from fpdf import FPDF


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Student Performance Mentor",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# GEMINI AI CONFIGURATION
# ==========================================
try:

    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

    genai.configure(api_key=GEMINI_API_KEY)

    model_ai = genai.GenerativeModel(
        "gemini-1.5-flash"
    )

    ai_available = True

except Exception as e:

    ai_available = False

    st.warning(
        "AI recommendations temporarily unavailable."
    )
# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================

with open("studentperformance.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================================
# TITLE
# ==========================================

st.title("📚 AI-Powered Student Performance Mentor")

st.markdown("""
This intelligent system can:

✅ Predict student performance  
✅ Analyze weak subjects  
✅ Generate AI academic recommendations  
✅ Translate recommendations into multiple languages  
✅ Convert recommendations to voice/audio  
✅ Generate downloadable PDF reports  
✅ Display performance analytics dashboard  
""")

# ==========================================
# LANGUAGE SELECTION
# ==========================================

language = st.selectbox(
    "🌍 Select Language",
    ["English", "French"]
)

language_codes = {
    "English": "en",
    "French": "fr"
}

selected_lang = language_codes[language]

# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("📥 Enter Student Information")

col1, col2 = st.columns(2)

with col1:

    study_hours_per_week = st.number_input(
        "Study Hours Per Week",
        min_value=0,
        max_value=40,
        value=5
    )

    attendance_rate = st.number_input(
        "Attendance Rate (%)",
        min_value=0,
        max_value=100,
        value=75
    )

    previous_grades = st.number_input(
        "Previous Grades",
        min_value=0,
        max_value=100,
        value=50
    )

with col2:

    participation = st.selectbox(
        "Participation in Extracurricular Activities",
        ["Yes", "No"]
    )

    parent_education_level = st.selectbox(
        "Parent Education Level",
        ["High School", "Bachelor", "Master", "Associate", "Doctorate"]
    )

# ==========================================
# SUBJECT SCORES
# ==========================================

st.subheader("📘 Subject Scores")

col3, col4, col5 = st.columns(3)

with col3:
    math_score = st.slider("Mathematics", 0, 100, 50)

with col4:
    english_score = st.slider("English", 0, 100, 50)

with col5:
    science_score = st.slider("Science", 0, 100, 50)

# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================

participation = 1 if participation == "Yes" else 0

parent_mapping = {
    "High School": 1,
    "Bachelor": 2,
    "Master": 3,
    "Associate": 4,
    "Doctorate": 5
}

parent_education_level = parent_mapping[parent_education_level]

# ==========================================
# PREPARE INPUT DATA
# ==========================================

input_data = np.array([[
    study_hours_per_week,
    attendance_rate,
    previous_grades,
    participation,
    parent_education_level
]])

# ==========================================
# PDF REPORT FUNCTION
# ==========================================

def generate_pdf(prediction_result, recommendation_text, weak_subjects_list):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=16)

    pdf.cell(
        200,
        10,
        txt="Student Performance Report",
        ln=True,
        align='C'
    )

    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Prediction: {prediction_result}", ln=True)

    pdf.cell(
        200,
        10,
        txt=f"Study Hours: {study_hours_per_week}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Attendance Rate: {attendance_rate}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Previous Grades: {previous_grades}",
        ln=True
    )

    pdf.ln(5)

    pdf.cell(200, 10, txt="Subject Scores", ln=True)

    pdf.cell(200, 10, txt=f"Mathematics: {math_score}", ln=True)

    pdf.cell(200, 10, txt=f"English: {english_score}", ln=True)

    pdf.cell(200, 10, txt=f"Science: {science_score}", ln=True)

    pdf.ln(5)

    pdf.cell(
        200,
        10,
        txt=f"Weak Subjects: {', '.join(weak_subjects_list)}",
        ln=True
    )

    pdf.ln(5)

    pdf.multi_cell(
        0,
        10,
        txt=f"AI Recommendations:\n\n{recommendation_text}"
    )

    filename = "student_report.pdf"

    pdf.output(filename)

    return filename

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔍 Predict Performance"):

    prediction = model.predict(input_data)[0]

    # ======================================
    # PREDICTION OUTPUT
    # ======================================

    if prediction == 1:
        st.success("✅ Student has high probability of PASSING")
        prediction_result = "PASS"
    else:
        st.error("❌ Student has high probability of FAILING")
        prediction_result = "FAIL"

    # ======================================
    # WEAK SUBJECT ANALYSIS
    # ======================================

    st.subheader("📚 Weak Subject Analysis")

    weak_subjects = []

    if math_score < 50:
        weak_subjects.append("Mathematics")

    if english_score < 50:
        weak_subjects.append("English")

    if science_score < 50:
        weak_subjects.append("Science")

    if weak_subjects:
        st.warning(
            f"Weak Subjects Identified: {', '.join(weak_subjects)}"
        )
    else:
        st.success("✅ No weak subjects identified.")

# ======================================
    # AI RECOMMENDATION ENGINE
    # ======================================

    st.subheader("🤖 AI Recommendations")

    # This variable is now safely inside the button click block
    prompt = f"""
    You are an intelligent academic mentor.
    Analyze this student's performance data and provide:
    - personalized academic advice
    - study improvement tips
    - motivation
    - weak subject intervention strategies

    Student Details:
    Study Hours Per Week: {study_hours_per_week}
    Attendance Rate: {attendance_rate}
    Previous Grades: {previous_grades}
    Mathematics Score: {math_score}
    English Score: {english_score}
    Science Score: {science_score}

    Prediction Result: {prediction_result}
    Weak Subjects: {weak_subjects}

    Keep the response concise and practical.
    """

    # --- AI Logic moved INSIDE the button block ---
    if ai_available:
        try:
            response = model_ai.generate_content(prompt)
            recommendation = response.text
        except Exception as e:
            st.error(f"AI Error: {e}")
            recommendation = "Focus on consistent study habits and attendance."
    else:
        recommendation = """
        Study consistently.
        Improve attendance.
        Focus more on weak subjects.
        Practice past questions regularly.
        """
    
    st.write(recommendation)

    # ======================================
    # TRANSLATION
    # ======================================

    if ai_available:
        translation_prompt = f"Translate the following text to {language}: {recommendation}"
        translation_response = model_ai.generate_content(translation_prompt)
        translated_text = translation_response.text
    else:
        translated_text = recommendation
    
    st.info(f"**Translated Recommendation ({language}):**")
    st.write(translated_text)

    # ... [Rest of your code for Audio, Dashboard, and PDF] ...

    # ======================================
    # TEXT TO SPEECH
    # ======================================

    try:

        tts = gTTS(
            text=translated_text,
            lang=selected_lang
        )

        audio_file = "recommendation.mp3"

        tts.save(audio_file)

        audio_bytes = open(audio_file, "rb").read()

        st.audio(audio_bytes, format="audio/mp3")

    except:
        st.warning("Audio generation unavailable.")

    # ======================================
    # PERFORMANCE DASHBOARD
    # ======================================

    st.markdown("---")

    st.subheader("📊 Student Performance Dashboard")

    risk_score = 100 - attendance_rate

    performance_score = (
        study_hours_per_week * 2 +
        attendance_rate * 0.4 +
        previous_grades * 0.6
    )

    col6, col7, col8 = st.columns(3)

    with col6:
        st.metric(
            "📚 Study Hours",
            f"{study_hours_per_week} hrs"
        )

    with col7:
        st.metric(
            "📝 Previous Grades",
            f"{previous_grades}%"
        )

    with col8:
        st.metric(
            "⚠️ Academic Risk",
            f"{risk_score}%"
        )

    # ======================================
    # BAR CHART
    # ======================================

    chart_data = pd.DataFrame({
        "Category": [
            "Study Hours",
            "Attendance",
            "Previous Grades"
        ],
        "Score": [
            study_hours_per_week,
            attendance_rate,
            previous_grades
        ]
    })

    fig_bar = px.bar(
        chart_data,
        x="Category",
        y="Score",
        text="Score",
        title="Academic Indicators"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # ======================================
    # PIE CHART
    # ======================================

    if performance_score >= 80:
        risk_label = "Low Risk"
        risk_value = 80

    elif performance_score >= 50:
        risk_label = "Medium Risk"
        risk_value = 50

    else:
        risk_label = "High Risk"
        risk_value = 20

    pie_data = pd.DataFrame({
        "Category": [risk_label, "Remaining"],
        "Value": [risk_value, 100 - risk_value]
    })

    fig_pie = px.pie(
        pie_data,
        names="Category",
        values="Value",
        title="Academic Risk Analysis"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # ======================================
    # TREND LINE CHART
    # ======================================

    trend_data = pd.DataFrame({
        "Weeks": [
            "Week 1",
            "Week 2",
            "Week 3",
            "Week 4"
        ],
        "Performance": [
            previous_grades - 10,
            previous_grades - 5,
            previous_grades,
            previous_grades + 5
        ]
    })

    fig_line = px.line(
        trend_data,
        x="Weeks",
        y="Performance",
        markers=True,
        title="Performance Trend"
    )

    st.plotly_chart(fig_line, use_container_width=True)

    # ======================================
    # RISK ANALYTICS
    # ======================================

    st.subheader("📈 Risk Analytics")

    if attendance_rate < 50:
        st.warning(
            "⚠️ Low attendance is affecting performance."
        )

    if study_hours_per_week < 5:
        st.warning(
            "⚠️ Study hours are below recommended level."
        )

    if previous_grades < 50:
        st.warning(
            "⚠️ Previous grades indicate academic risk."
        )

    if (
        attendance_rate >= 70 and
        study_hours_per_week >= 10 and
        previous_grades >= 60
    ):
        st.success(
            "✅ Student shows strong academic potential."
        )

    # ======================================
    # PDF REPORT DOWNLOAD
    # ======================================

    pdf_file = generate_pdf(
        prediction_result,
        recommendation,
        weak_subjects
    )

    with open(pdf_file, "rb") as file:

        st.download_button(
            label="📥 Download Student Report",
            data=file,
            file_name="student_report.pdf",
            mime="application/pdf"
        )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.write("AI-Powered Student Performance Mentor")

st.write("Developed by Hikma Yahya")
