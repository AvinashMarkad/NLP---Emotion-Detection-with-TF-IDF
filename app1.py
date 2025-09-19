import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🎭 Emotion Detection with TF-IDF + Logistic Regression (Preloaded Dataset)")

# --- Load dataset directly from train.txt ---
@st.cache_data
def load_data():
    df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])
    return df

df = load_data()

st.write("### Sample Data")
st.dataframe(df.head())

# --- Vectorization (Improved TF-IDF) ---
texts = df["text"]
labels = df["emotion"]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams
    min_df=2,             # ignore rare words
    max_df=0.9,           # ignore very frequent words
    sublinear_tf=True,    # scale term frequency
    stop_words="english"  # remove common stopwords
)
X = vectorizer.fit_transform(texts)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.20, random_state=42, stratify=labels
)

# --- Train Classifier (Improved Logistic Regression) ---
clf = LogisticRegression(
    C=5, 
    class_weight="balanced", 
    max_iter=2000, 
    solver="lbfgs", 
    multi_class="auto"
)
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"✅ Model trained with accuracy: **{acc:.2%}**")

# Classification report
st.subheader("📊 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion matrix
st.subheader("🔎 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
fig, ax = plt.subplots()
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=clf.classes_, yticklabels=clf.classes_, ax=ax
)
st.pyplot(fig)

# --- User input for prediction ---
st.subheader("✍️ Test Your Own Sentence")
user_input = st.text_area("Enter text here:")

if st.button("Predict"):
    if user_input.strip() != "":
        X_input = vectorizer.transform([user_input])
        pred = clf.predict(X_input)[0]
        proba = clf.predict_proba(X_input)[0]

        st.write(f"### 🎯 Predicted Emotion: **{pred}**")

        # Show probabilities in %
        proba_df = pd.DataFrame([proba * 100], columns=clf.classes_)
        st.write("### 📈 Probabilities (%)")
        st.dataframe(proba_df.round(2))
