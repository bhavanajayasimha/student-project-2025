from flask import Flask, render_template, request
import joblib
import html

app = Flask(__name__)

model = joblib.load("fake_news_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

def predict_news(text: str):
    text = text.strip()
    if len(text) < 20:
        return {"label": "UNKNOWN", "reason": "Please enter a longer news-like sentence.", "confidence": None}

    safe_text = html.escape(text)
    X = tfidf.transform([safe_text])
    pred = int(model.predict(X)[0])

    try:
        prob = model.predict_proba(X)[0]
        confidence = float(max(prob))
    except Exception:
        confidence = None

    label = "FAKE" if pred == 0 else "REAL"
    return {"label": label, "confidence": confidence}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("news_text", "")
        res = predict_news(user_text)

        if res["label"] == "UNKNOWN":
            message = res["reason"]
        else:
            result = res

    return render_template("index.html", result=result, message=message, user_text=user_text)

if __name__ == "__main__":
    app.run(debug=True)
