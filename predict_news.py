import joblib

# Load saved model and TF-IDF vectorizer
model = joblib.load("fake_news_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

def predict_news(text: str) -> str:
    """Return 'FAKE' or 'REAL' for a given news text."""
    text_tfidf = tfidf.transform([text])
    pred = model.predict(text_tfidf)[0]
    return "FAKE" if pred == 0 else "REAL"

if __name__ == "__main__":
    while True:
        user_input = input("\nEnter news text (or type 'quit' to stop): ")
        if user_input.lower() == "quit":
            break
        print("Prediction:", predict_news(user_input))
