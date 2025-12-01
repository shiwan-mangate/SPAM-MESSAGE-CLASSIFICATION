# 📧 SMS SPAM DETECTION SYSTEM  
💡 **Spam Message Classifier using NLP & Machine Learning**  
An end-to-end Natural Language Processing project that classifies SMS messages as **Spam** or **Ham** using TF-IDF vectorization and multiple ML models.

---

## 🖼️ App Screenshot  
*(Add your Streamlit / Web App screenshot here)*

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

**Dataset Used:** SMS Spam Collection Dataset  
- **Total Messages:** 5572  
- **Cleaned Final Messages:** 5169  
- **Target Labels:**  
  - `0` → Ham (legitimate message)  
  - `1` → Spam (fraud/scam/promo)

### 🔧 Engineered Features  

| Feature | Description |
|--------|-------------|
| `num_char` | Character count of SMS |
| `num_words` | Word count |
| `num_sent` | Number of sentences |
| `transformed_text` | Cleaned, tokenized & stemmed text |

---

## 🛠️ Tech Stack & Tools  

### **Languages & Libraries**
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
- Removed duplicate records  
- Encoded labels  
- Prepared clean dataset  

### ✔ 2. Text Processing (`transform_text()`)
- Lowercasing  
- Tokenization  
- Removing punctuation & non-alphanumeric words  
- Stopword removal  
- Stemming using **Snowball Stemmer**  

---

## 📊 Exploratory Data Analysis (EDA)

### 📈 **Visualizations Included**
- Spam vs Ham **bar plot**  
- **KDE plots** (characters & words distribution)  
- **Pairplot** of engineered features  
- **Heatmap** (feature correlations)  
- **WordClouds** (spam & ham)  
- **Top 30 most common words** (spam & ham)  

**Insights:**  
Spam messages tend to be longer and contain high-frequency promotional keywords.

---

## 🔡 Feature Extraction (TF-IDF)

### **TF-IDF Vectorization Used**
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

### ✔ Why TF-IDF?

TF-IDF (Term Frequency – Inverse Document Frequency) helps the model understand which words are important. It works by reducing the weight of very common words and increasing the importance of meaningful ones.

**Benefits of TF-IDF in Spam Detection:**

- Prevents common words like **“the”** from dominating  
- Highlights important spam-indicative words like:  
  **“free”, “win”, “congratulations”**  
- Improves classification accuracy  
- Works extremely well with **Multinomial Naive Bayes (MNB)**  
- Reduces noise in text data and enhances signal  

---

## 🤖 Machine Learning Models Tried

A variety of ML models were trained and evaluated for performance:

| Algorithm | Accuracy | Precision |
|----------|----------|-----------|
| **KNN** | 0.9052 | 1.0000 |
| **Naive Bayes (MNB)** | 0.9729 | 1.0000 |
| **Extra Trees Classifier** | 0.9797 | 0.9756 |
| **SVC** | 0.9758 | 0.9747 |
| **Random Forest** | 0.9748 | 0.9746 |
| **Logistic Regression** | 0.9555 | 0.9693 |
| **XGBoost** | 0.9680 | 0.9565 |
| **Gradient Boosting** | 0.9477 | 0.9375 |
| **Bagging Classifier** | 0.9574 | 0.8615 |
| **AdaBoost** | 0.9245 | 0.8409 |
| **Decision Tree** | 0.9323 | 0.8400 |

---

## 🏆 Final Model: **Multinomial Naive Bayes (MNB)**  

The best-performing model based on evaluation:

- **Accuracy:** 97.29%  
- **Precision:** 100% (Perfect spam detection)  
- Lightweight & extremely fast  
- Ideal for text classification  
- Highly interpretable and widely used in NLP tasks  

---

## 💾 Model Saving

```python
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(mnb, open("model.pkl", "wb"))

