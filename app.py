import streamlit as st
import joblib
import re

st.set_page_config(page_title="MediBot - Symptom Checker", page_icon="🩺", layout="centered")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .main-title { text-align: center; color: white; font-size: 42px; font-weight: bold; }
    .subtitle { text-align: center; color: #f0f0f0; font-size: 16px; margin-bottom: 30px; }
    .chat-bubble-user {
        background-color: #4CAF50; color: white; padding: 12px 18px;
        border-radius: 18px 18px 0px 18px; margin: 10px 0; max-width: 80%;
        margin-left: auto; font-size: 16px;
    }
    .chat-bubble-bot {
        background-color: white; color: #333; padding: 12px 18px;
        border-radius: 18px 18px 18px 0px; margin: 10px 0; max-width: 80%;
        font-size: 16px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = joblib.load('disease_prediction_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, tfidf, le

model, tfidf, le = load_models()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_disease(symptom_text):
    cleaned = clean_text(symptom_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = probabilities[prediction] * 100
    disease = le.inverse_transform([prediction])[0]
    return disease, confidence

st.markdown('<div class="main-title">🩺 MediBot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tell me your symptoms, and I\'ll suggest a possible condition</div>', unsafe_allow_html=True)

st.markdown("""
    <div style="background-color: #fff3cd; color: #856404; padding: 15px;
    border-radius: 10px; border: 2px solid #ffc107; margin-bottom: 20px;
    font-size: 15px; text-align: center;">
    ⚠️ This is an educational tool, not a medical diagnosis. Please consult a real doctor for actual health concerns.
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">{msg["text"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-bot">{msg["text"]}</div>', unsafe_allow_html=True)

user_input = st.chat_input("Type your symptoms here... (e.g. I have fever and headache)")

if user_input:
    st.session_state.messages.append({"role": "user", "text": user_input})
    disease, confidence = predict_disease(user_input)
    reply = f"Based on your symptoms, this might be **{disease}** (confidence: {confidence:.1f}%). Please consult a doctor to confirm."
    st.session_state.messages.append({"role": "bot", "text": reply})
    st.rerun()
