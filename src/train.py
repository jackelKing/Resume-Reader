import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from utils import clean_text

# Load dataset
df = pd.read_csv('data/Resume.csv')

# Assuming dataset has columns: 'Resume' and 'Category'
df['Cleaned'] = df['Resume'].apply(clean_text)

X = df['Cleaned']
y = df['Category']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train classifier
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Test
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save models
joblib.dump(clf, 'models/role_clf.joblib')
joblib.dump(vectorizer, 'models/vectorizer.joblib')
print("Models saved to models/")
