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
# TRANSLATIONS DICTIONARY
# English and Hausa for every UI string
# ==========================================

TRANSLATIONS = {
    "English": {
        # Page title & intro
        "page_title":            "📚 AI-Powered Student Performance Mentor",
        "intro":                 (
            "This intelligent system can:\n\n"
            "✅ Predict student performance  \n"
            "✅ Analyze weak subjects  \n"
            "✅ Generate AI academic recommendations  \n"
            "✅ Translate recommendations into multiple languages  \n"
            "✅ Convert recommendations to voice/audio  \n"
            "✅ Generate downloadable PDF reports  \n"
            "✅ Display performance analytics dashboard  "
        ),

        # Language selector
        "select_language":       "🌍 Select Language",

        # Input section
        "enter_info":            "📥 Enter Student Information",
        "study_hours":           "Study Hours Per Week",
        "attendance_rate":       "Attendance Rate (%)",
        "previous_grades":       "Previous Grades",
        "participation":         "Participation in Extracurricular Activities",
        "participation_yes":     "Yes",
        "participation_no":      "No",
        "parent_edu":            "Parent Education Level",
        "edu_high_school":       "High School",
        "edu_bachelor":          "Bachelor",
        "edu_master":            "Master",
        "edu_associate":         "Associate",
        "edu_doctorate":         "Doctorate",

        # Subject scores
        "subject_scores":        "📘 Subject Scores",
        "mathematics":           "Mathematics",
        "english":               "English",
        "science":               "Science",

        # Prediction button & results
        "predict_btn":           "🔍 Predict Performance",
        "pass_msg":              "✅ Student has high probability of PASSING",
        "fail_msg":              "❌ Student has high probability of FAILING",
        "pass_result":           "PASS",
        "fail_result":           "FAIL",

        # Weak subject labels (used in analysis & PDF)
        "weak_math":             "Mathematics",
        "weak_english":          "English",
        "weak_science":          "Science",

        # AI recommendation section
        "ai_reco_header":        "🤖 AI Recommendations",
        "default_reco":          "Study consistently and focus on weak subjects.",
        "ai_error_msg":          "AI is currently taking a break. Here is some general advice:",
        "translation_label":     "Translation:",

        # Audio
        "audio_unavailable":     "Audio unavailable.",

        # Dashboard
        "dashboard_header":      "📊 Student Performance Dashboard",
        "metric_study":          "📚 Study Hours",
        "metric_study_unit":     "hrs",
        "metric_grades":         "📝 Previous Grades",
        "metric_risk":           "⚠️ Academic Risk",

        # Chart labels
        "bar_cat_study":         "Study Hours",
        "bar_cat_attendance":    "Attendance",
        "bar_cat_grades":        "Previous Grades",
        "bar_title":             "Academic Indicators",
        "bar_x":                 "Category",
        "bar_y":                 "Score",

        # Pie chart
        "pie_title":             "Academic Risk Analysis",
        "pie_low_risk":          "Low Risk",
        "pie_medium_risk":       "Medium Risk",
        "pie_high_risk":         "High Risk",
        "pie_remaining":         "Remaining",

        # Trend chart
        "trend_title":           "Performance Trend",
        "trend_x":               "Weeks",
        "trend_y":               "Performance",
        "trend_week1":           "Week 1",
        "trend_week2":           "Week 2",
        "trend_week3":           "Week 3",
        "trend_week4":           "Week 4",

        # Risk analytics
        "risk_header":           "📈 Risk Analytics",
        "warn_attendance":       "⚠️ Low attendance is affecting performance.",
        "warn_study":            "⚠️ Study hours are below recommended level.",
        "warn_grades":           "⚠️ Previous grades indicate academic risk.",
        "success_potential":     "✅ Student shows strong academic potential.",

        # PDF report
        "pdf_title":             "Student Performance Report",
        "pdf_prediction":        "Prediction",
        "pdf_study_hours":       "Study Hours",
        "pdf_attendance":        "Attendance Rate",
        "pdf_prev_grades":       "Previous Grades",
        "pdf_subject_scores":    "Subject Scores",
        "pdf_math":              "Mathematics",
        "pdf_english":           "English",
        "pdf_science":           "Science",
        "pdf_weak_subjects":     "Weak Subjects",
        "pdf_ai_reco":           "AI Recommendations",
        "pdf_download_btn":      "📥 Download Student Report",

        # Footer
        "footer_title":          "AI-Powered Student Performance Mentor",
        "footer_author":         "Developed by Hikma Yahya",

        # AI prompt language instruction
        "ai_prompt_lang":        "English",
        # gTTS language code
        "tts_lang":              "en",
    },

    "Hausa": {
        # Page title & intro
        "page_title":            "📚 Tsarin Koyarwa na AI don Dalibai",
        "intro":                 (
            "Wannan tsarin hankali zai iya:\n\n"
            "✅ Hasashen aikin dalibai  \n"
            "✅ Nazarin fannonin da suke da rauni  \n"
            "✅ Samar da shawarwarin karatu na AI  \n"
            "✅ Fassara shawarwari zuwa harsuna daban-daban  \n"
            "✅ Canza shawarwari zuwa murya/sauti  \n"
            "✅ Samar da rahotannin PDF da za a iya zazzagewa  \n"
            "✅ Nuna allon nazarin nasarar karatu  "
        ),

        # Language selector
        "select_language":       "🌍 Zaɓi Harshe",

        # Input section
        "enter_info":            "📥 Shigar da Bayanan Dalibai",
        "study_hours":           "Awannin Karatu a Mako",
        "attendance_rate":       "Yawan Zuwa Makaranta (%)",
        "previous_grades":       "Maki na Baya",
        "participation":         "Shiga Ayyukan Bayan Makaranta",
        "participation_yes":     "Eh",
        "participation_no":      "A'a",
        "parent_edu":            "Matakin Ilimin Iyaye",
        "edu_high_school":       "Sakandare",
        "edu_bachelor":          "Digiri na Farko",
        "edu_master":            "Digiri na Biyu",
        "edu_associate":         "Takardar Shaida",
        "edu_doctorate":         "Digiri na Uku",

        # Subject scores
        "subject_scores":        "📘 Maki na Fannonin Karatu",
        "mathematics":           "Lissafi",
        "english":               "Turanci",
        "science":               "Kimiyya",

        # Prediction button & results
        "predict_btn":           "🔍 Hasashen Nasara",
        "pass_msg":              "✅ Dalibai na da yiwuwar WUCEWA a jarrabawa",
        "fail_msg":              "❌ Dalibai na da yiwuwar KASA a jarrabawa",
        "pass_result":           "WUCE",
        "fail_result":           "KASA",

        # Weak subject labels
        "weak_math":             "Lissafi",
        "weak_english":          "Turanci",
        "weak_science":          "Kimiyya",

        # AI recommendation section
        "ai_reco_header":        "🤖 Shawarwarin AI",
        "default_reco":          "Ka yi karatu kullum kuma ka mai da hankali kan fannonin da suke da rauni.",
        "ai_error_msg":          "AI yana hutu a yanzu. Ga wasu shawarwari na gaba ɗaya:",
        "translation_label":     "Fassara:",

        # Audio
        "audio_unavailable":     "Sauti ba ya samuwa.",

        # Dashboard
        "dashboard_header":      "📊 Allon Nasarar Karatu na Dalibai",
        "metric_study":          "📚 Awannin Karatu",
        "metric_study_unit":     "awanni",
        "metric_grades":         "📝 Maki na Baya",
        "metric_risk":           "⚠️ Hadarin Karatu",

        # Chart labels
        "bar_cat_study":         "Awannin Karatu",
        "bar_cat_attendance":    "Zuwa Makaranta",
        "bar_cat_grades":        "Maki na Baya",
        "bar_title":             "Abubuwan Karatu",
        "bar_x":                 "Nau'i",
        "bar_y":                 "Maki",

        # Pie chart
        "pie_title":             "Nazarin Haɗarin Karatu",
        "pie_low_risk":          "Ƙarancin Haɗari",
        "pie_medium_risk":       "Matsakaicin Haɗari",
        "pie_high_risk":         "Babban Haɗari",
        "pie_remaining":         "Sauran",

        # Trend chart
        "trend_title":           "Ci Gaban Nasara",
        "trend_x":               "Makonni",
        "trend_y":               "Nasara",
        "trend_week1":           "Mako na 1",
        "trend_week2":           "Mako na 2",
        "trend_week3":           "Mako na 3",
        "trend_week4":           "Mako na 4",

        # Risk analytics
        "risk_header":           "📈 Nazarin Haɗari",
        "warn_attendance":       "⚠️ Ƙarancin zuwa makaranta yana shafar nasara.",
        "warn_study":            "⚠️ Awannin karatu sun ƙasa da matakin da ake ba da shawara.",
        "warn_grades":           "⚠️ Maki na baya suna nuna haɗarin karatu.",
        "success_potential":     "✅ Dalibai na da ƙarfin karatu mai ƙarfi.",

        # PDF report
        "pdf_title":             "Rahoto na Nasarar Dalibai",
        "pdf_prediction":        "Hasashe",
        "pdf_study_hours":       "Awannin Karatu",
        "pdf_attendance":        "Yawan Zuwa Makaranta",
        "pdf_prev_grades":       "Maki na Baya",
        "pdf_subject_scores":    "Maki na Fannonin Karatu",
        "pdf_math":              "Lissafi",
        "pdf_english":           "Turanci",
        "pdf_science":           "Kimiyya",
        "pdf_weak_subjects":     "Fannonin da ke da Rauni",
        "pdf_ai_reco":           "Shawarwarin AI",
        "pdf_download_btn":      "📥 Zazzage Rahoto na Dalibai",

        # Footer
        "footer_title":          "Tsarin Koyarwa na AI don Dalibai",
        "footer_author":         "An ƙirƙira ta Hikma Yahya",

        # AI prompt language instruction
        "ai_prompt_lang":        "Hausa",
        # gTTS does not have a native Hausa code; fall back to English TTS
        # but the text shown in the UI will still be Hausa
        "tts_lang":              "en",
    },
}

# ==========================================
# GEMINI AI CONFIGURATION
# ==========================================

ai_available = False
try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model_ai = genai.GenerativeModel("gemini-1.5-flash-latest")
        ai_available = True
    else:
        st.error("API Key missing in secrets!")
except Exception as e:
    st.warning(f"AI Setup Error: {e}")

# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================

with open("studentperformance.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================================
# LANGUAGE SELECTION  (placed first so all
# labels below react to the chosen language)
# ==========================================

language = st.selectbox(
    "🌍 Select Language / Zaɓi Harshe",
    ["English", "Hausa"]
)

# Shorthand accessor
T = TRANSLATIONS[language]

# ==========================================
# TITLE
# ==========================================

st.title(T["page_title"])
st.markdown(T["intro"])

# ==========================================
# INPUT SECTION
# ==========================================

st.subheader(T["enter_info"])

col1, col2 = st.columns(2)

with col1:
    study_hours_per_week = st.number_input(
        T["study_hours"],
        min_value=0,
        max_value=40,
        value=5
    )

    attendance_rate = st.number_input(
        T["attendance_rate"],
        min_value=0,
        max_value=100,
        value=75
    )

    previous_grades = st.number_input(
        T["previous_grades"],
        min_value=0,
        max_value=100,
        value=50
    )

with col2:
    participation = st.selectbox(
        T["participation"],
        [T["participation_yes"], T["participation_no"]]
    )

    parent_education_level = st.selectbox(
        T["parent_edu"],
        [
            T["edu_high_school"],
            T["edu_bachelor"],
            T["edu_master"],
            T["edu_associate"],
            T["edu_doctorate"],
        ]
    )

# ==========================================
# SUBJECT SCORES
# ==========================================

st.subheader(T["subject_scores"])

col3, col4, col5 = st.columns(3)

with col3:
    math_score = st.slider(T["mathematics"], 0, 100, 50)

with col4:
    english_score = st.slider(T["english"], 0, 100, 50)

with col5:
    science_score = st.slider(T["science"], 0, 100, 50)

# ==========================================
# ENCODE CATEGORICAL VARIABLES
# Map translated labels back to numeric values
# ==========================================

participation_encoded = 1 if participation == T["participation_yes"] else 0

edu_mapping = {
    T["edu_high_school"]: 1,
    T["edu_bachelor"]:    2,
    T["edu_master"]:      3,
    T["edu_associate"]:   4,
    T["edu_doctorate"]:   5,
}
parent_education_encoded = edu_mapping[parent_education_level]

# ==========================================
# PREPARE INPUT DATA
# ==========================================

input_data = np.array([[
    study_hours_per_week,
    attendance_rate,
    previous_grades,
    participation_encoded,
    parent_education_encoded
]])

# ==========================================
# PDF REPORT FUNCTION
# ==========================================

def generate_pdf(prediction_result, recommendation_text, weak_subjects_list, t):
    """Generate a fully translated PDF report."""

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=t["pdf_title"], ln=True, align="C")
    pdf.ln(10)

    # Prediction & basic info
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"{t['pdf_prediction']}: {prediction_result}", ln=True)
    pdf.cell(200, 10, txt=f"{t['pdf_study_hours']}: {study_hours_per_week}", ln=True)
    pdf.cell(200, 10, txt=f"{t['pdf_attendance']}: {attendance_rate}", ln=True)
    pdf.cell(200, 10, txt=f"{t['pdf_prev_grades']}: {previous_grades}", ln=True)
    pdf.ln(5)

    # Subject scores
    pdf.cell(200, 10, txt=t["pdf_subject_scores"], ln=True)
    pdf.cell(200, 10, txt=f"{t['pdf_math']}: {math_score}", ln=True)
    pdf.cell(200, 10, txt=f"{t['pdf_english']}: {english_score}", ln=True)
    pdf.cell(200, 10, txt=f"{t['pdf_science']}: {science_score}", ln=True)
    pdf.ln(5)

    # Weak subjects
    pdf.cell(
        200, 10,
        txt=f"{t['pdf_weak_subjects']}: {', '.join(weak_subjects_list) if weak_subjects_list else '-'}",
        ln=True
    )
    pdf.ln(5)

    # AI recommendation
    # FPDF uses Latin-1; encode to handle special characters gracefully
    safe_reco = recommendation_text.encode("latin-1", errors="replace").decode("latin-1")
    pdf.multi_cell(0, 10, txt=f"{t['pdf_ai_reco']}:\n\n{safe_reco}")

    filename = "student_report.pdf"
    pdf.output(filename)
    return filename


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button(T["predict_btn"]):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(T["pass_msg"])
        prediction_result = T["pass_result"]
    else:
        st.error(T["fail_msg"])
        prediction_result = T["fail_result"]

    # --- WEAK SUBJECT ANALYSIS ---
    weak_subjects = []
    if math_score    < 50: weak_subjects.append(T["weak_math"])
    if english_score < 50: weak_subjects.append(T["weak_english"])
    if science_score < 50: weak_subjects.append(T["weak_science"])

    # --- AI RECOMMENDATION ENGINE ---
    st.subheader(T["ai_reco_header"])

    recommendation  = T["default_reco"]
    translated_text = recommendation

    if ai_available:
        try:
            # Ask Gemini to respond directly in the chosen language
            prompt = (
                f"Student Performance: {prediction_result}. "
                f"Weak subjects: {weak_subjects}. "
                f"Suggest 3 specific study tips. "
                f"Reply entirely in {T['ai_prompt_lang']}."
            )
            response       = model_ai.generate_content(prompt)
            recommendation = response.text

            # If English was chosen, no separate translation needed;
            # for Hausa, translate the English recommendation into Hausa
            if language == "English":
                translated_text = recommendation
                st.write(recommendation)
            else:
                translation_prompt = (
                    f"Translate the following text to {T['ai_prompt_lang']} "
                    f"(Hausa language used in Nigeria):\n\n{recommendation}"
                )
                t_response      = model_ai.generate_content(translation_prompt)
                translated_text = t_response.text

                st.write(recommendation)
                st.info(f"**{T['translation_label']}**")
                st.write(translated_text)

        except Exception as e:
            st.warning(T["ai_error_msg"])
            st.write(recommendation)
            print(f"DEBUG ERROR: {e}")
    else:
        st.write(recommendation)

    # ======================================
    # TEXT TO SPEECH
    # ======================================
    try:
        tts = gTTS(text=translated_text, lang=T["tts_lang"])
        audio_file = "recommendation.mp3"
        tts.save(audio_file)
        st.audio(audio_file, format="audio/mp3")
    except Exception:
        st.warning(T["audio_unavailable"])

    # ======================================
    # PERFORMANCE DASHBOARD
    # ======================================

    st.markdown("---")
    st.subheader(T["dashboard_header"])

    risk_score = 100 - attendance_rate

    performance_score = (
        study_hours_per_week * 2 +
        attendance_rate      * 0.4 +
        previous_grades      * 0.6
    )

    col6, col7, col8 = st.columns(3)

    with col6:
        st.metric(
            T["metric_study"],
            f"{study_hours_per_week} {T['metric_study_unit']}"
        )

    with col7:
        st.metric(
            T["metric_grades"],
            f"{previous_grades}%"
        )

    with col8:
        st.metric(
            T["metric_risk"],
            f"{risk_score}%"
        )

    # ======================================
    # BAR CHART
    # ======================================

    chart_data = pd.DataFrame({
        T["bar_x"]: [
            T["bar_cat_study"],
            T["bar_cat_attendance"],
            T["bar_cat_grades"],
        ],
        T["bar_y"]: [
            study_hours_per_week,
            attendance_rate,
            previous_grades,
        ]
    })

    fig_bar = px.bar(
        chart_data,
        x=T["bar_x"],
        y=T["bar_y"],
        text=T["bar_y"],
        title=T["bar_title"]
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # ======================================
    # PIE CHART
    # ======================================

    if performance_score >= 80:
        risk_label = T["pie_low_risk"]
        risk_value = 80
    elif performance_score >= 50:
        risk_label = T["pie_medium_risk"]
        risk_value = 50
    else:
        risk_label = T["pie_high_risk"]
        risk_value = 20

    pie_data = pd.DataFrame({
        "Category": [risk_label, T["pie_remaining"]],
        "Value":    [risk_value, 100 - risk_value]
    })

    fig_pie = px.pie(
        pie_data,
        names="Category",
        values="Value",
        title=T["pie_title"]
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # ======================================
    # TREND LINE CHART
    # ======================================

    trend_data = pd.DataFrame({
        T["trend_x"]: [
            T["trend_week1"],
            T["trend_week2"],
            T["trend_week3"],
            T["trend_week4"],
        ],
        T["trend_y"]: [
            previous_grades - 10,
            previous_grades - 5,
            previous_grades,
            previous_grades + 5,
        ]
    })

    fig_line = px.line(
        trend_data,
        x=T["trend_x"],
        y=T["trend_y"],
        markers=True,
        title=T["trend_title"]
    )

    st.plotly_chart(fig_line, use_container_width=True)

    # ======================================
    # RISK ANALYTICS
    # ======================================

    st.subheader(T["risk_header"])

    if attendance_rate < 50:
        st.warning(T["warn_attendance"])

    if study_hours_per_week < 5:
        st.warning(T["warn_study"])

    if previous_grades < 50:
        st.warning(T["warn_grades"])

    if (
        attendance_rate      >= 70 and
        study_hours_per_week >= 10 and
        previous_grades      >= 60
    ):
        st.success(T["success_potential"])

    # ======================================
    # PDF REPORT DOWNLOAD
    # ======================================

    pdf_file = generate_pdf(
        prediction_result,
        translated_text,    # PDF uses the displayed language text
        weak_subjects,
        T
    )

    with open(pdf_file, "rb") as file:
        st.download_button(
            label=T["pdf_download_btn"],
            data=file,
            file_name="student_report.pdf",
            mime="application/pdf"
        )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.write(T["footer_title"])
st.write(T["footer_author"])
