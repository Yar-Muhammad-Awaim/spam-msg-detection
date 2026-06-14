import joblib
import pandas as pd
from pre_processing import pre_process_msg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

model = MultinomialNB()

le = LabelEncoder()

df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = pre_process_msg(df, "v2")

# Fit and transform the v1 column containing spam and ham using label encoder
y = le.fit_transform(df["v1"])


X_train, X_test, y_train, y_test = train_test_split(
    df["v2"], y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()

# Fit and transfrom training data, while only transfrom the test data.
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model.fit(X_train, y_train)

# Save model to the file
joblib.dump(model, "model/spam_msg_detector.joblib")

# Save label encoder to the file
joblib.dump(le, "model/label_encoder.joblib")

# Save tf-idf vectorizer to the file
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")

# Predict using test data
y_pred = model.predict(X_test)

# Check the prediction score against the actual data
score_accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of my model is {score_accuracy * 100} %.")

print("Detailed Classification Report: ")
print(classification_report(y_test, y_pred, target_names=le.classes_))
