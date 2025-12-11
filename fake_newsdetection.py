import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. Load the data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# 2. Add a label column: 0 = fake, 1 = real
fake["label"] = 0
true["label"] = 1

# 3. Combine datasets
data = pd.concat([fake, true], axis=0).reset_index(drop=True)

# 4. Keep only the text (title + text) and label
data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")
data = data[["content", "label"]]

print("Total samples:", len(data))
print(data.head())
# 5. Split data into features (X) and labels (y)
X = data["content"]      # text
y = data["label"]        # 0 = fake, 1 = real

# 6. Train–test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # so results are same every time
    stratify=y           # keep fake/real ratio same in train & test
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))
# 7. Create TF-IDF vectorizer to convert text into numeric features
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)

# Learn vocabulary from training data and transform it
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform test data (use same vocabulary as train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF test shape :", X_test_tfidf.shape)
# 8. Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 9. Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# 10. Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# 11. Save the trained model and TF-IDF vectorizer
joblib.dump(model, "fake_news_model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
print("\nSaved model and vectorizer to disk.")

# 11. Simple function to test new text
def predict_news(news_text: str):
    text_tfidf = tfidf.transform([news_text])
    prediction = model.predict(text_tfidf)[0]
    return "FAKE" if prediction == 0 else "REAL"

if __name__ == "__main__":
    while True:
        user_input = input("\nEnter news text (or type 'quit' to stop): ")
        if user_input.lower() == "quit":
            break
        print("Prediction:", predict_news(user_input))
