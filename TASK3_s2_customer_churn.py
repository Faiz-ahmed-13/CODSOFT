import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv("C:/Users/Faiz Ahmed/OneDrive/Desktop/CODSOFT/TASK3_churnpred/Churn_Modelling.csv")
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# One-hot encode
df_encoded = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True, dtype=int)

# Split features/target
X = df_encoded.drop('Exited', axis=1)
y = df_encoded['Exited']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale continuous features
scaler = StandardScaler()
num_cols = ['CreditScore', 'Age', 'Balance', 'Tenure', 'EstimatedSalary']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Calculate class weights for imbalanced data
class_weights = len(y_train[y_train==0]) / len(y_train[y_train==1])

def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """Helper function to evaluate models consistently"""
    # Fit the model
    model.fit(X_train, y_train)
    
    # Train predictions
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else [0]*len(X_train)
    
    # Test predictions
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(X_test)
    
    # Print metrics
    print("\n" + "="*50)
    print(model_name)
    print("="*50)
    
    print("\nTraining Performance:")
    print(f"Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    if hasattr(model, "predict_proba"):
        print(f"ROC AUC: {roc_auc_score(y_train, train_prob):.4f}")
    print(classification_report(y_train, train_pred))
    
    print("\nTest Performance:")
    print(f"Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    if hasattr(model, "predict_proba"):
        print(f"ROC AUC: {roc_auc_score(y_test, test_prob):.4f}")
    print(classification_report(y_test, test_pred))
    
    # Return fitted model and probabilities for PR curve
    return model, test_prob

# Initialize PR curve plot
plt.figure(figsize=(8, 6))

# 1. Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', solver='saga', max_iter=1000, random_state=42)
lr_model, lr_prob = evaluate_model(lr_model, "Logistic Regression", X_train, y_train, X_test, y_test)

# 2. Random Forest
rf_model = RandomForestClassifier(
    class_weight={0:1, 1:class_weights}, 
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42
)
rf_model, rf_prob = evaluate_model(rf_model, "Random Forest", X_train, y_train, X_test, y_test)

# 3. XGBoost
xgb_model = XGBClassifier(
    scale_pos_weight=class_weights,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='aucpr',
    random_state=42,
    use_label_encoder=False
)
xgb_model, xgb_prob = evaluate_model(xgb_model, "XGBoost", X_train, y_train, X_test, y_test)

# Plot PR curves
for name, prob in [('Logistic Regression', lr_prob), 
                   ('Random Forest', rf_prob), 
                   ('XGBoost', xgb_prob)]:
    precision, recall, _ = precision_recall_curve(y_test, prob)
    plt.plot(recall, precision, label=name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Feature Importance for best model
print("\nFeature Importance (XGBoost):")
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance)

# Threshold optimization for XGBoost
print("\nThreshold Optimization for XGBoost:")
precision, recall, thresholds = precision_recall_curve(y_test, xgb_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"Optimal F1-Score: {f1_scores[optimal_idx]:.4f}")

# Predict with optimal threshold
y_pred_optimal = (xgb_prob >= optimal_threshold).astype(int)
print("\nOptimized Classification Report:")
print(classification_report(y_test, y_pred_optimal))