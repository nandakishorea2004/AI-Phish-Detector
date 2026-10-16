
import streamlit as st
import joblib

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="AI Email Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #1d4ed8);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .result-card {
        padding: 22px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-top: 14px;
        margin-bottom: 14px;
    }

    .result-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .muted {
        color: #64748b;
        font-size: 0.95rem;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("phishing_model.joblib")


try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.code(str(error))
    st.stop()


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🛡️ AI-Powered Email Classifier</h1>
    <p>Analyze email text using a machine-learning model.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("About the project")
    st.write(
        "This application uses TF-IDF text features and "
        "a Logistic Regression classifier."
    )

    st.divider()

    st.subheader("Model information")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Text features:** TF-IDF")
    st.write("**Training dataset:** 82,486 emails")

    st.divider()

    st.caption(
        "Predictions reflect patterns learned from the dataset. "
        "They are not a guarantee of email safety."
    )


# ---------------- INPUT SECTION ----------------
st.subheader("Analyze an email")
st.write(
    "Paste the subject and body of an email in the box below."
)

with st.form("email_analysis_form"):
    email_text = st.text_area(
        "Email content",
        height=230,
        placeholder=(
            "Subject: Example subject\n\n"
            "Paste the email message here..."
        )
    )

    submitted = st.form_submit_button(
        "🔍 Analyze Email",
        type="primary",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------
if submitted:
    if not email_text.strip():
        st.warning("Please enter some email text before analyzing.")
    else:
        prediction = int(model.predict([email_text])[0])

        probabilities = model.predict_proba([email_text])[0]
        classes = list(model.classes_)

        predicted_index = classes.index(prediction)
        confidence = float(probabilities[predicted_index])

        st.divider()
        st.subheader("Analysis results")

        # Neutral class names until dataset labels are verified
        if prediction == 0:
            result_title = "Class 0"
            result_icon = "🔵"
            result_description = (
                "The model assigned this email to dataset class 0."
            )
        else:
            result_title = "Class 1"
            result_icon = "🟣"
            result_description = (
                "The model assigned this email to dataset class 1."
            )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    {result_icon} Predicted result: {result_title}
                </div>
                <div class="muted">{result_description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence indicator
        st.subheader("Model confidence")
        st.metric(
            label="Confidence in the predicted class",
            value=f"{confidence * 100:.2f}%"
        )

        st.progress(confidence)

        st.caption(
            "This is the model's estimated probability for its chosen "
            "class, not a guarantee that the prediction is correct."
        )

        # Show probabilities for both classes
        st.subheader("Class probability breakdown")

        col0, col1 = st.columns(2)

        for class_value, probability in zip(classes, probabilities):
            with (col0 if class_value == 0 else col1):
                st.markdown(f"**Class {class_value}**")
                st.write(f"{probability * 100:.2f}%")
                st.progress(float(probability))

        with st.expander("View submitted email"):
            st.text(email_text)

        st.info(
            "Interpretation note: Class 0 and Class 1 are kept neutral "
            "until the dataset's label definitions are confirmed."
        )


# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "AI Email Classifier | Academic machine-learning project | "
    "Use predictions as decision support, not as a security guarantee."
)
