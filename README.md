# 📧 SMS SPAM DETECTION SYSTEM  
💡 **Spam Message Classifier using NLP & Machine Learning**  
An end-to-end Natural Language Processing project that classifies SMS messages as **Spam** or **Ham** using TF-IDF vectorization and a variety of ML models.

---

## 🖼️ App Screenshot  
(Add your Streamlit / Web App screenshot here)

---

## 🚀 Motivation  
Spam messages are increasing rapidly—from fake offers to phishing attempts.  
This project builds an **intelligent ML-powered spam detector** to:

- Identify malicious or promotional SMS  
- Reduce fraud risks  
- Help telecom systems maintain message quality  
- Provide accurate, automated message classification  

---

## 📂 Dataset Overview  

Dataset Used: **SMS Spam Collection Dataset**  
- **Total Messages:** 5572  
- **Cleaned Final Messages:** 5169  
- **Target Labels:**  
  - `0` → Ham (legitimate message)  
  - `1` → Spam (fraud/scam/promo)

### Engineered Features  
| Feature | Description |
|--------|-------------|
| `num_char` | Character count of SMS |
| `num_words` | Word count |
| `num_sent` | Number of sentences |
| `transformed_text` | Cleaned, tokenized & stemmed text |

---

## 🛠️ Tech Stack & Tools  

### **Programming & Libraries**
- Python  
- Pandas, NumPy  
- NLTK (tokenizer, stopwords, stemming)  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost  
- WordCloud  
- Pickle  

---

## 🧹 Data Preprocessing & NLP Pipeline  

### ✔ 1. Data Cleaning  
- Dropped irrelevant columns  
- Removed missing values  
- Removed duplicates  
- Encoded labels  
- Prepared clean dataset  

### ✔ 2. Text Processing (`transform_text()`)
- Lowercasing  
- Tokenization  
- Removing punctuation & non-alphanumeric words  
- Stopword removal  
- Stemming using Snowball Stemmer  

---

## 📊 Exploratory Data Analysis (EDA)

### 📈 **Visualizations Included:**
- Spam vs Ham bar plot  
- KDE Plots (characters & words)  
- Pairplot  
- Heatmap  
- WordClouds (spam & ham)  
- Top 30 most common words (spam & ham)  

These insights show spam messages tend to be longer and contain specific promotional/fraud indicators.

---

## 🔡 Feature Extraction (TF-IDF)

### **TF-IDF Vectorization**

```python
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

✔ Why TF-IDF?

Prevents common words like “the” from dominating

Highlights meaningful words like “free”, “win”, “congratulations”

Improves classification accuracy

Works extremely well with Multinomial Naive Bayes

Reduces noise in text data

🤖 Machine Learning Models Tried
Algorithm	Accuracy	Precision
KNN	0.9052	1.0000
Naive Bayes (MNB)	0.9729	1.0000
Extra Trees Classifier	0.9797	0.9756
SVC	0.9758	0.9747
Random Forest	0.9748	0.9746
Logistic Regression	0.9555	0.9693
XGBoost	0.9680	0.9565
Gradient Boosting	0.9477	0.9375
Bagging Classifier	0.9574	0.8615
AdaBoost	0.9245	0.8409
Decision Tree	0.9323	0.8400
🏆 Final Model: Multinomial Naive Bayes (MNB)

Best-performing model with:

Accuracy: 97.29%

Precision: 100% (Perfect spam detection)

Lightweight & fast

Ideal for text classification

Highly interpretable

💾 Model Saving
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(mnb, open("model.pkl", "wb"))

🌐 Streamlit Web App

🚀 Live SMS Spam Classification Web App:
👉 https://spam-message-classification-by-shiwan.streamlit.app/

This web app allows users to input any message and instantly detect whether it is Spam or Ham.

📦 GitHub Repository

📂 Complete Project Code Available Here:
👉 https://github.com/shiwan-mangate/SPAM-MESSAGE-CLASSIFICATION

This includes:

Full Jupyter Notebook

Preprocessing scripts

TF-IDF + Model pickle files

Streamlit app code

README & documentation

🧠 Professional Use Cases

This system can be integrated into:

📱 Mobile SMS filtering
📡 Telecom automation systems
🔐 Fraud & phishing detection
💬 Customer message pre-screening
📨 Promotional message classification

🚧 Future Improvements

Add deep learning models (LSTM, BERT)

Add multilingual support

Add SHAP/LIME explainability

Deploy REST API

Build a mobile-friendly UI

👨‍💻 Author

Shiwan Mangate
B.Tech in Artificial Intelligence
NIT Rourkela
