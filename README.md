# 🏦 Creditworthiness Prediction App

A machine learning web application that predicts whether an individual is a good or bad credit risk based on 24 financial and personal attributes. This project uses a Random Forest model trained on the UCI German Credit dataset and is deployed using Streamlit for easy interaction.

---

## 📌 Key Features

- ✅ Predicts creditworthiness (Good/Bad Credit Risk)
- 🔍 Uses 24 numeric features including credit amount, age, duration, employment, etc.
- 🌲 Trained using Random Forest with cost-sensitive learning to handle class imbalance
- 🧠 Model trained and saved using `scikit-learn==1.7.1` for safe, consistent deployment
- 🌐 Frontend powered by Streamlit for live user input and real-time predictions

---

## 🧠 Dataset Used

- **Source**: [UCI German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Format**: Numeric version with 1000 samples and 24 features + target class
- **Target**: `1 = Good Credit Risk`, `0 = Bad Credit Risk`

---

## 🧪 Project Structure

Creditworthiness/
├── data/ # Dataset (german.data-numeric)
├── model/ # Saved model (credit_model.pkl) and features
├── app.py # Streamlit web application
├── train_model.py # ML training script
├── requirements.txt # Package dependencies
└── README.md # Project documentation



---

## ⚙️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/creditworthiness-predictor.git
cd creditworthiness-predictor

python -m venv venv
venv\Scripts\activate   # On Windows

pip install -r requirements.txt

python train_model.py

streamlit run app.py
```

---

## 📬 Author

- **Ritik Sotwal**
- 4th Year, Electronics and Computer Engineering, MBM University