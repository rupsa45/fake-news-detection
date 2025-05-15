from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Only required the first time
nltk.download('stopwords')

app = FastAPI()

# Load the saved model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
stop_words = set(stopwords.words('english'))

# Request body format
class NewsInput(BaseModel):
    text: str

# Preprocess function (same as before)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# Root route
@app.get("/")
def read_root():
    return {"message": "Fake News Detection API is running"}

# Prediction endpoint
@app.post("/")
def predict_news(data: NewsInput):
    cleaned = clean_text(data.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    result = "Real" if prediction == 1 else "Fake"
    return {"prediction": result}
