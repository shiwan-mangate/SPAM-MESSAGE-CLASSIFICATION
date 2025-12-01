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
