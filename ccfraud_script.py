import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#loading the dataset
train_set = pd.read_csv("C:/Users/Faiz Ahmed/OneDrive/Desktop/CODSOFT/TASK2_ccfraud/fraudTrain.csv")
test_set = pd.read_csv("C:/Users/Faiz Ahmed/OneDrive/Desktop/CODSOFT/TASK2_ccfraud/fraudTest.csv")
print(train_set.head())
print(test_set.head())
columns_to_drop = ['trans_num', 'unix_time', 'first', 'last']
train_set.drop(columns=columns_to_drop,axis=1, inplace=True)
test_set.drop(columns=columns_to_drop, axis = 1,inplace=True)
print(train_set.head())
print(test_set.head())

X_train = train_set.drop('is_fraud',axis=1)
Y_train = train_set['is_fraud']
X_test = test_set.drop('is_fraud', axis=1)
Y_test = test_set['is_fraud']

#categorical columns to be converted to numerical
categorical_cols = [col for col in X_train.select_dtypes(include=['object']).columns if col != 'trans_date_trans_time']
numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Convert 'trans_date_trans_time' to datetime and extract features
if 'trans_date_trans_time' in X_train.columns:
    # Convert 'trans_date_trans_time' to datetime format
    X_train['trans_date_trans_time'] = pd.to_datetime(X_train['trans_date_trans_time'])
    X_test['trans_date_trans_time'] = pd.to_datetime(X_test['trans_date_trans_time'])

    # Extract useful features from 'trans_date_trans_time'
    X_train['hour'] = X_train['trans_date_trans_time'].dt.hour
    X_train['day'] = X_train['trans_date_trans_time'].dt.day
    X_train['month'] = X_train['trans_date_trans_time'].dt.month

    X_test['hour'] = X_test['trans_date_trans_time'].dt.hour
    X_test['day'] = X_test['trans_date_trans_time'].dt.day
    X_test['month'] = X_test['trans_date_trans_time'].dt.month

    # Drop the original 'trans_date_trans_time' column
    X_train.drop('trans_date_trans_time', axis=1, inplace=True, errors='ignore')
    X_test.drop('trans_date_trans_time', axis=1, inplace=True, errors='ignore')

label_encoder = LabelEncoder()
for col in categorical_cols:
    # Fit LabelEncoder on the union of train and test data to avoid unseen labels
    combined = pd.concat([X_train[col].fillna('missing'), X_test[col].fillna('missing')], ignore_index=True)
    label_encoder.fit(combined.astype(str))

    # Transform both train and test data
    X_train[col] = label_encoder.transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])

# Scale numerical features using StandardScaler
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])  # Use transform (not fit_transform) for test data

# Ensure X_train and X_test have the same columns
X_train, X_test = X_train.align(X_test, join='inner', axis=1)

# Print the final set of columns to verify
print("Final set of columns in X_train and X_test:")
print(X_train.columns)

model = RandomForestClassifier(n_estimators = 100,random_state=42)
model.fit(X_train,Y_train)

X_train_pred = model.predict(X_train)
X_test_pred = model.predict(X_test)
print (X_train_pred)
print (X_test_pred)

confusion_matrix_result = confusion_matrix(Y_test, X_test_pred)
print("Confusion Matrix: ", confusion_matrix_result)
accuracy = accuracy_score(Y_test, X_test_pred)
print("Accuracy: ", accuracy)
classification_report_result = classification_report(Y_test, X_test_pred)
print("Classification Report: ", classification_report_result)


