💳 Credit Card Fraud Detection (Machine Learning + XAI)

A complete end-to-end credit card fraud detection system built using Machine Learning, Explainable AI (SHAP), and a Streamlit web application.

This project detects fraudulent transactions using a Random Forest model and explains why a prediction was made using SHAP (SHapley values) — the gold standard for explainable AI.

🚀 Features
✔ Machine Learning Model

RandomForestClassifier

Handles imbalanced dataset (undersampling)

Scaling of Time & Amount

High performance:

Accuracy: ~96%

Fraud Recall: ~90%

AUC: ~0.98

✔ Explainable AI (SHAP)

SHAP summary plot (dot plot)

Shows which features contributed to fraud

Auto-saves as: shap_summary.png

✔ Visualizations

ROC Curve

Precision–Recall Curve

✔ Web Application (Streamlit)

Random real transaction prediction

Manual feature input mode

Fraud probability (%)

Clean UI

Uses trained model (fraud_model.pkl)

📁 Project Structure
fraud-detection/
│── fraud_advanced.py        # Main ML model (training + SHAP + metrics)
│── fraud_app.py             # Streamlit web application
│── creditcard.csv           # Dataset (NOT pushed to GitHub)
│── fraud_model.pkl          # Saved ML model
│── shap_summary.png         # SHAP output image
│── README.md
│── .gitignore

📊 Dataset

Kaggle Dataset: Credit Card Fraud Detection

284,807 rows

30 anonymized PCA features (V1–V28)

Highly imbalanced (fraud = 0.17%)

Dataset contains:

Time — seconds elapsed

Amount — transaction amount

V1–V28 — PCA transformed features

Class — 1 = Fraud, 0 = Safe

🧠 How the Model Works

Load dataset

Scale Time and Amount

Split into train/test

Undersample majority class

Train RandomForest

Evaluate using:

Confusion Matrix

Classification Report

AUC

Create explainability graph using SHAP

Save model as .pkl

🏗️ Running the ML Model

Install dependencies:

pip install -r requirements.txt


Train + generate ROC/PR/SHAP:

python fraud_advanced.py


Outputs:

ROC curve

Precision–Recall curve

shap_summary.png

fraud_model.pkl

🌐 Running the Streamlit App
streamlit run fraud_app.py


Browser window opens automatically:

Two modes:
✔ Random Transaction (Recommended)

Loads a real transaction

Shows prediction + probability

Shows actual dataset label

✔ Manual Input (Advanced)

Enter Time, Amount, V1–V28

Useful for experimenting

📌 Sample SHAP Output

shap_summary.png shows:

Red / Blue dots

Feature importance (V14, V10, V4…)

How each feature pushed the model towards FRAUD or SAFE

This makes the model fully transparent and explainable.

🏁 Conclusion

This project is a complete ML pipeline:

Fraud detection

Imbalanced learning

Explainable AI

Streamlit web app

Industry-standard workflow

Perfect for:

College Major Project

Resume project

Portfolio

ML/AI Interviews

👨‍💻 Author

omShukla69
A complete Machine Learning + Explainable AI Project