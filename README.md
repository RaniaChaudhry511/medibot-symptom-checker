# 🩺 MediBot Symptom Checker


https://github.com/user-attachments/assets/0f1476ac-235a-4a46-b7f8-9bc2918773d6




A symptom checker that predicts a possible disease from symptoms typed by the user. Built with machine learning and deployed as a Streamlit web app.

**Live app:** https://medibot-symptom-checker-aaf24fbr3owhs7cmkgk6k4.streamlit.app/

## Demo
A short screen recording of the app is available.

In the demo, the symptoms `increased thirst, frequent urination, blurred vision, fatigue, weight loss` are entered and MediBot suggests **diabetes** (confidence 44%), which was the correct prediction for these symptoms.

## How It Works

1. The user types their symptoms as text.
2. The text is converted to numbers using a TF-IDF vectorizer.
3. The trained model predicts the most likely disease.
4. The label encoder converts the prediction back into the disease name.

## Files
- `app.py`: Streamlit app
- `disease_prediction_model.pkl`: trained model
- `tfidf_vectorizer.pkl`: text vectorizer
- `label_encoder.pkl`: maps predictions to disease names
- `requirements.txt`: dependencies

## Tech Stack

Python, Scikit-learn, Pandas, NumPy, Streamlit

## How to Run Locally

```bash
git clone https://github.com/RaniaChaudhry511/medibot-symptom-checker.git
cd medibot-symptom-checker
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Limitations

- This project was made **for practice and learning purposes only**.
  
- The dataset was **not very large**, so the model **can predict wrong results**.
  
- **Predictions depend on the symptoms entered.** If you enter incorrect or incomplete symptoms, the result can be wrong.
  
- Please **do not enter only common symptoms** like flu, cough or body pain, because these appear in many different diseases and the model cannot tell them apart. Enter as many **specific symptoms** as possible.

## Disclaimer

This project is **not medical advice** and must not be used for diagnosis. Please consult a qualified doctor for any health concern.

## Author

Rania Chaudhry
