import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.sparse import hstack
data = pd.read_csv("emails.csv")
def has_url(text):
    return 1 if re.search(r'http[s]?://', str(text)) else 0
data['has_url'] = data['text'].apply(has_url)
data['label'] = data['label'].map({'safe': 0, 'phishing': 1})
data.dropna(inplace=True)
vectorizer = TfidfVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(data['text'])
X = hstack([X_text, data[['has_url']].values])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, zero_division=0))
def classify_email(email_text):
    url_feature = 1 if re.search(r'http[s]?://', str(email_text)) else 0
    text_vector = vectorizer.transform([email_text])
    final_input = hstack([text_vector, [[url_feature]]])
    prediction = model.predict(final_input)[0]
    return "Phishing" if prediction == 1 else "Safe"
email1 = "Win a free iPhone now! Click http://scam.com"
email2 = "Reminder: Team meeting at 10 AM tomorrow"
print("\nEmail 1 Prediction:", classify_email(email1))
print("Email 2 Prediction:", classify_email(email2))