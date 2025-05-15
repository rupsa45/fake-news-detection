# ML training script


# train_model.py
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

# Load and label data
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")
fake_df['label'] = 0
real_df['label'] = 1

data = pd.concat([fake_df, real_df], axis=0).sample(frac=1).reset_index(drop=True)
data['content'] = data['title'] + " " + data['text']

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = [word for word in text.split() if word not in stop_words]
    return ' '.join(text)

data['content'] = data['content'].apply(clean_text)

X = data['content']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
