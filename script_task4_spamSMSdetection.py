import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load and preprocess data
df = pd.read_csv("C:/Users/Faiz Ahmed/OneDrive/Desktop/CODSOFT/TASK4_spamSMS/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

# Encode labels (spam=1, ham=0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Text preprocessing
def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())

df['text'] = df['text'].apply(preprocess)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Helper function to evaluate models"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Get metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Metric': ['Accuracy', 
                  'Precision (Spam)', 
                  'Recall (Spam)', 
                  'F1-Score (Spam)',
                  'Confusion Matrix'],
        'Value': [
            f"{accuracy:.4f}",
            f"{report['1']['precision']:.4f}",
            f"{report['1']['recall']:.4f}",
            f"{report['1']['f1-score']:.4f}",
            str(cm)
        ]
    })
    
    print(f"\n{model_name} Evaluation:")
    print(results)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, results

# Evaluate Naive Bayes
nb_model, nb_results = evaluate_model(
    MultinomialNB(), 
    X_train, y_train, X_test, y_test,
    "Naive Bayes"
)

# Evaluate Logistic Regression
lr_model, lr_results = evaluate_model(
    LogisticRegression(max_iter=1000), 
    X_train, y_train, X_test, y_test,
    "Logistic Regression"
)

# Evaluate SVM
svm_model, svm_results = evaluate_model(
    SVC(kernel='linear'),  # Using linear kernel for better performance on text data
    X_train, y_train, X_test, y_test,
    "Support Vector Machine"
)

# Combine results for comparison
comparison_df = pd.DataFrame({
    'Metric': nb_results['Metric'],
    'Naive Bayes': nb_results['Value'],
    'Logistic Regression': lr_results['Value'],
    'SVM': svm_results['Value']
})

print("\nModel Comparison:")
print(comparison_df)



