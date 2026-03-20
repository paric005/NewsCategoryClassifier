import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="News Classifier", layout="wide")

# -----------------------------
# GLOBAL DYNAMIC BACKGROUND 🔥
# -----------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #1f4037, #99f2c8, #00c6ff, #0072ff);
        background-size: 400% 400%;
        animation: gradientBG 10s ease infinite;
        color: white;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    h1, h2, h3, h4 {
        color: #ffffff;
    }

    .stButton>button {
        background-color: #ff4b2b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
    }

    .stTextArea textarea {
        border-radius: 10px;
    }

    .card {
        background: rgba(0,0,0,0.4);
        padding: 20px;
        border-radius: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/model.pkl")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/combined.csv")
df = df.dropna()
df["text"] = df["headlines"] + " " + df["description"]

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("", ["EDA", "Prediction", "Metrics"])

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<h1 style='text-align:center;'>📰 AI News Classifier</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# EDA TAB
# -----------------------------
if page == "EDA":
    st.markdown("## 📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Category Distribution")
        st.bar_chart(df["category"].value_counts())
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📏 Text Length Distribution")
    df["length"] = df["text"].apply(len)
    st.line_chart(df["length"])
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# PREDICTION TAB
# -----------------------------
elif page == "Prediction":
    st.markdown("## 🔮 Predict News Category")

    # Categories Box
    st.markdown("""
        <div class='card'>
        <h4>📌 Available Categories:</h4>
        <ul>
            <li>🏏 Sports</li>
            <li>💼 Business</li>
            <li>🎬 Entertainment</li>
            <li>🎓 Education</li>
            <li>💻 Technology</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✍️ Enter your news text:")

    user_input = st.text_area("", height=150)

    if st.button("🚀 Predict"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text")
        else:
            prediction = model.predict([user_input])
            probability = model.predict_proba([user_input])

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🎯 Result")
            st.success(f"Category: {prediction[0]}")
            st.info(f"Confidence Score: {max(probability[0]):.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# METRICS TAB
# -----------------------------
elif page == "Metrics":
    st.markdown("## 📈 Model Performance")

    X = df["text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("✅ Accuracy", f"{acc:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Classification Report")
        st.text(classification_report(y_test, y_pred))
        st.markdown("</div>", unsafe_allow_html=True)