import pandas as pd
import re, time, joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print("1. Loading dataset...")
df = pd.read_csv('data/IMDB Dataset.csv')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>|[^a-zA-Z]', ' ', text).lower()
    return ' '.join([stemmer.stem(w) for w in text.split() if w not in stop_words])

print("2. Preprocessing text (50,000 samples)...")
df['clean_review'] = df['review'].apply(clean_text)

y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values

print("3. Vectorizing with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "linear_svm": CalibratedClassifierCV(LinearSVC(dual='auto', random_state=42))
}

benchmarks = {}

print("\n4. Training and Benchmarking Models...")
for name, clf in models.items():
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_duration = time.perf_counter() - t0
    
    t_inf = time.perf_counter()
    y_pred = clf.predict(X_test)
    inference_time = (time.perf_counter() - t_inf) / len(y_test) * 1000  # ms per sample
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    benchmarks[name] = {
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "train_time_sec": round(train_duration, 2),
        "latency_ms_per_sample": round(inference_time, 4)
    }
    print(f"-> {name.upper()}: Accuracy={benchmarks[name]['accuracy']}% | F1={benchmarks[name]['f1_score']}")
    joblib.dump(clf, f"{name}_model.pkl")

# Update saved vectorizer and benchmark metadata
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(benchmarks, 'model_benchmarks.pkl')
print("\nAll models and benchmark metrics successfully serialized!")