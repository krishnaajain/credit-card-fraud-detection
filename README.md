

# 💳 Credit Card Fraud Detection System

A robust and interactive web application built with **Streamlit** that uses advanced **machine learning techniques** to detect fraudulent credit card transactions in real-time. This project leverages PCA-transformed data, feature scaling, SMOTE for class imbalance, and a Random Forest classifier to deliver high-accuracy predictions.

---

## 🚀 Live Demo

> 🟢 This is a local application. To run it, follow the setup instructions below.

---

## 🧠 Features

- 🧮 Predicts the probability of a transaction being fraudulent
- 📊 Accepts PCA features (V1–V28), Amount, and Hour of transaction
- 📈 Displays the top 5 most influential features used in the prediction
- 🧪 Optimized thresholding for better fraud classification using F1-score
- 🖥️ Interactive and intuitive UI with Streamlit

---

## 📁 Project Structure

```

.
├── app.py                       # Streamlit application
├── creditcard.py               # Full preprocessing, training, and evaluation script
├── fraud\_detection\_model.pkl   # Trained RandomForest model
├── scaler.pkl                  # Scaler for amount and hour features
├── optimal\_threshold.pkl       # Optimized classification threshold
├── creditcard.csv              # Dataset (not included here)
└── README.md                   # Project documentation

````



## 🧪 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features**:
  - V1 to V28: PCA-transformed features
  - Amount: Transaction amount
  - Time: Seconds elapsed since the first transaction (used to derive `Hour`)
  - Class: 0 (Genuine), 1 (Fraudulent)

---

## 🛠️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/fraud-detection-app.git
   cd fraud-detection-app


2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

4. **Access the App**
   Open the local URL (e.g., `http://localhost:8501`) in your browser.


## 🏗️ Model Building Summary

* Removed all rows with missing values
* Derived `Hour` feature from `Time`
* Scaled `Amount` and `Hour` using `StandardScaler`
* Balanced class distribution using **SMOTE**
* Trained a **RandomForestClassifier** with `class_weight='balanced_subsample'`
* Tuned threshold using **F1-score optimization**
* Saved model, scaler, and threshold using `joblib`

---

## 📊 Performance Metrics

| Metric             | Value                            |
| ------------------ | -------------------------------- |
| Accuracy           | High                             |
| ROC AUC Score      | \~0.98                           |
| Precision & Recall | Optimized using custom threshold |
| Feature Importance | Displayed in app UI              |

---

## 📂 Important Files

* `fraud_detection_model.pkl`: Trained Random Forest model
* `scaler.pkl`: Standard scaler for Amount and Hour
* `optimal_threshold.pkl`: Custom threshold for prediction
* `app.py`: Streamlit frontend to run prediction

---

## 🧾 Requirements

```txt
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
joblib
streamlit
```

To install:

```bash
pip install -r requirements.txt
```

---

## 💡 Future Improvements

* Integrate with real-time payment APIs
* Add user authentication and role management
* Provide historical fraud trend dashboard
* Enable batch transaction uploads and batch predictions


---

## 📄 License

This project is open-source and available under the MIT License.

```

---

Let me know if you want a markdown version with a preview image, contributor guidelines, or deployment instructions for platforms like Streamlit Cloud or Hugging Face Spaces.
```
