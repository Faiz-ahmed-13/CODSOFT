# CODSOFT :TASK 1 :


# 🎬 Movie Genre Classification

This project focuses on building a machine learning model that can classify the **genre of a movie** based on its **title and plot description**. The system uses natural language processing (NLP) techniques and classic machine learning algorithms to predict genres effectively.

---

## 🧠 Task

**Goal:** Predict the genre of a movie from its plot summary and title using machine learning models.

---

## 📂 Dataset

The dataset contains the following columns:

- `TITLE`: The title of the movie.
- `DESCRIPTION`: A brief plot summary of the movie.
- `GENRE`: The target label — the movie's genre.

Train and test CSV files are used:
- `output_dataset.csv` – for training/validation.
- `out_dataset.csv` – for testing.
- `outsol_dataset.csv` – ground truth for test set.

---

## ⚙️ Models Used

1. **Naive Bayes (GaussianNB)**
2. **Logistic Regression**
3. **Support Vector Machine (SVM - LinearSVC)**

TF-IDF vectorization was applied to the combined movie title and description to extract textual features.

---

## 🔁 Workflow

1. **Preprocessing:**
   - Combined `TITLE` and `DESCRIPTION` into a single `text` column.
   - Applied TF-IDF vectorization (max features = 5000).

2. **Training:**
   - Models were trained on 80% of the data and validated on 20%.

3. **Evaluation:**
   - Accuracy scores were computed for each model on both training and test splits.
   - Final predictions were compared with the ground truth labels.

---

## 📊 Results

| Model               | Train Accuracy | Test Accuracy |
|--------------------|----------------|---------------|
| GaussianNB          | 0.656          | 0.419         |
| Logistic Regression | 0.986          | 0.825         |
| Linear SVM          | 1.000          | 0.826         |

> The SVM and Logistic Regression models significantly outperformed Naive Bayes in accuracy.

---


## 🚀 How to Run

1. Clone this repository.
2. Install required Python packages:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Place the dataset files in the same directory.
4. Run the Python script:
   ```bash
   python cstask1sc2_mvgenre.py
   ```

---

## 📈 Final Test Evaluation

A separate test set was evaluated using the Logistic Regression model. Accuracy is printed at the end of the script along with actual vs. predicted genre comparison.

---

## 📌 Notes

- Ensure that the dataset file paths are correctly set before running the script.
- This project can be further improved by experimenting with deep learning models like LSTM or transformers.

---

## 📬 Contact

For any queries or suggestions, feel free to reach out!

--- 
Based on the script and video you provided for **Task 2: Credit Card Fraud Detection**, here's a fully prepared `README.md` section that you can copy and paste into your existing README:

---
# CODSOFT :TASK 2 :


# 💳 Credit Card Fraud Detection

This project involves detecting fraudulent credit card transactions using machine learning models. The dataset contains transactional data and the goal is to classify each transaction as either **fraudulent** or **legitimate**.

---

## 🧠 Task

**Goal:** Build a model that detects fraudulent credit card transactions by analyzing patterns in the transaction features. Algorithms such as Logistic Regression, Decision Trees, or Random Forests can be applied for classification.

---

## 📂 Dataset

The dataset used in this task includes:

- `fraudTrain.csv` – training set with labeled transactions.
- `fraudTest.csv` – test set with labeled transactions.

Key columns include:
- Transaction amount, category, merchant, time, and location.
- Target column: `is_fraud` (1 = Fraudulent, 0 = Legitimate)

---

## ⚙️ Model Used

- **Random Forest Classifier**

Data preprocessing steps included:
- Dropping irrelevant columns such as `trans_num`, `unix_time`, and personal names.
- Encoding categorical variables with `LabelEncoder`.
- Extracting `hour`, `day`, and `month` from the transaction timestamp.
- Standardizing numerical features using `StandardScaler`.

---

## 🔁 Workflow

1. **Preprocessing:**
   - Date-time features extracted and redundant columns dropped.
   - Categorical data encoded.
   - Data normalized using standard scaling.

2. **Training:**
   - Model trained using `RandomForestClassifier(n_estimators=100)`.

3. **Evaluation:**
   - Accuracy, confusion matrix, and classification report computed on the test set.

---

## 📊 Results

| Metric         | Value        |
|----------------|--------------|
| Accuracy       | 0.974        |
| Precision (1)  | 0.90         |
| Recall (1)     | 0.87         |
| F1-Score (1)   | 0.88         |

> The model performs very well with high accuracy and strong fraud detection performance

---

## 🚀 How to Run

1. Clone this repository.
2. Install required Python packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Make sure the dataset files (`fraudTrain.csv`, `fraudTest.csv`) are in the working directory.
4. Run the Python script:
   ```bash
   python csTASK2_ccfraud_script.py
   ```

---

## 📌 Notes

- For better fraud detection, consider techniques like SMOTE for handling class imbalance, or try neural network models for more advanced learning.
- Real-time fraud detection systems often require model deployment and integration with transaction streams.

---

## 📬 Contact

For feedback or collaboration, feel free to connect!

---
Based on your script and video for **Task 3: Customer Churn Prediction**, here is a ready-to-use `README.md` section that summarizes your work and can be pasted directly into your GitHub documentation.

---

# CODSOFT :TASK 3 :

# 🔁 Customer Churn Prediction

This project focuses on building a machine learning model to predict customer churn for a subscription-based service. By analyzing customer behavior and demographic features, the system aims to identify users who are likely to stop using the service.

---

## 🧠 Task

**Goal:** Predict whether a customer will churn using historical data, including features such as tenure, credit score, account balance, and geographic/demographic attributes.

---

## 📂 Dataset

Dataset used: `Churn_Modelling.csv`

Key columns:
- `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- Target column: `Exited` (1 = Churned, 0 = Active)

---

## ⚙️ Models Used

Three models were trained and evaluated:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

Key preprocessing steps:
- Dropped irrelevant identifiers: `RowNumber`, `CustomerId`, `Surname`
- Encoded categorical variables (`Geography`, `Gender`) using one-hot encoding
- Scaled numerical columns using `StandardScaler`
- Applied class weighting to handle class imbalance

---

## 🔁 Workflow

1. **Data Preprocessing**
2. **Training with Class Weighting**
3. **Model Evaluation**:
   - Accuracy
   - ROC AUC Score
   - Precision, Recall, and F1-Score
4. **Precision-Recall Curve Analysis**
5. **Threshold Optimization for XGBoost**
6. **Feature Importance (XGBoost)**

---

## 📊 Results

### Test Set Performance

| Model              | Accuracy | ROC AUC | Optimized F1 (XGB) |
|--------------------|----------|---------|--------------------|
| Logistic Regression| 0.81     | 0.85    | —                  |
| Random Forest      | 0.86     | 0.91    | —                  |
| XGBoost            | 0.87     | 0.92    | 0.84               |

> The XGBoost model provided the best performance overall, especially after optimizing the decision threshold based on F1-score.

---


## 🚀 How to Run

1. Clone this repository.
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib xgboost
   ```
3. Ensure `Churn_Modelling.csv` is placed in the script directory.
4. Run the script:
   ```bash
   python TASK3_s2_customer_churn.py
   ```

---

## 📌 Notes

- This project uses **XGBoost's `scale_pos_weight`** to handle class imbalance and optimizes the decision threshold for improved F1-score.
- Consider integrating SHAP or LIME for better interpretability of predictions in production.

---

## 📬 Contact

For questions, suggestions, or contributions, feel free to reach out!

---

Based on your script, task description, and video for **Task 4: Spam SMS Detection**, here is a polished and detailed `README.md` section you can paste directly into your GitHub repo:

---

# CODSOFT :TASK 4 :

# 📩 Spam SMS Detection

This project focuses on building a machine learning model to classify SMS messages as **spam** or **legitimate (ham)**. It applies natural language processing (NLP) techniques such as TF-IDF and uses multiple classifiers to evaluate model performance.

---

## 🧠 Task

**Goal:** Build an AI model that can classify SMS messages as spam or legitimate using textual data. Algorithms such as Naive Bayes, Logistic Regression, or Support Vector Machines are used for this classification task.

---

## 📂 Dataset

Dataset used: [`spam.csv`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

Key columns:
- `v1`: The label (`spam` or `ham`)
- `v2`: The actual message text

The dataset is preprocessed as follows:
- Renamed `v1` to `label` and `v2` to `text`
- Labels encoded (`spam` → 1, `ham` → 0)
- Lowercased and stripped punctuation

---

## ⚙️ Models Used

1. **Multinomial Naive Bayes**
2. **Logistic Regression**
3. **Support Vector Machine (Linear Kernel)**

All models use **TF-IDF vectorization** with `max_features=5000` to transform text into numerical feature vectors.

---

## 🔁 Workflow

1. **Text Preprocessing**
   - Convert to lowercase
   - Remove punctuation
   - Token normalization

2. **Vectorization**
   - TF-IDF applied to message text

3. **Model Training**
   - Data split: 80% training, 20% testing
   - Models evaluated using:
     - Accuracy
     - Precision / Recall / F1 (for spam class)
     - Confusion Matrix

4. **Results Aggregation**
   - All results compiled in a comparison DataFrame

---

## 📊 Results

| Metric              | Naive Bayes | Logistic Regression | SVM         |
|---------------------|-------------|----------------------|-------------|
| Accuracy            | 0.9780      | 0.9810               | 0.9810      |
| Precision (Spam)    | 0.9634      | 0.9756               | 0.9756      |
| Recall (Spam)       | 0.9151      | 0.9151               | 0.9151      |
| F1-Score (Spam)     | 0.9387      | 0.9444               | 0.9444      |
| Confusion Matrix    | [[949   0],<br> [ 20 146]] | [[949   0],<br> [ 14 152]] | [[949   0],<br> [ 14 152]] |

> Logistic Regression and SVM achieved the highest overall accuracy and F1-score, showing strong performance in spam classification.

---

## 🎥 Output Demo

A video demonstrating data loading, training, evaluation, and result comparison is available below:

📽️ **[Watch the Demo Video](task4_spamSMSdetection_output.mp4)**

> ⚠️ If this file is too large to upload directly to GitHub, consider uploading it to YouTube or Google Drive and updating the link accordingly.

---

## 🚀 How to Run

1. Clone the repository.
2. Install required packages:
   ```bash
   pip install pandas scikit-learn
   ```
3. Place `spam.csv` in the working directory.
4. Run the Python script:
   ```bash
   python task4_spam_sms_detection.py
   ```

---

## 📌 Notes

- Consider using **n-grams** or **word embeddings** (e.g., Word2Vec or GloVe) to improve results.
- For deployment, wrap the model in a web or mobile application with live message scanning.

---

## 📬 Contact

For feedback or collaboration, feel free to reach out!


---

# 📝 Task 5: Handwritten Text Generation using RNN

This project implements a **character-level recurrent neural network (RNN)** to generate handwritten-like text. It is trained on sequences of pen stroke data paired with character labels to learn realistic handwriting patterns.

## 📌 Project Highlights

- Uses a **Bidirectional LSTM** for capturing forward and backward dependencies in handwriting patterns.
- Trains on stroke sequences from real handwritten data.
- Supports generation of new handwriting from any input text string.
- Includes robust training loop with gradient clipping, dropout, and learning rate scheduling.
- Efficient batching and padding for variable-length sequences.

## 📁 Dataset

The model expects a dataset in the form of a `.npz` file named:
```
deepwriting_training.npz
```

This file must contain:
- `strokes`: List of stroke sequences (arrays of shape `[T, 3]` where T is the number of timesteps)
- `char_labels`: Corresponding character indices
- `alphabet`: List of characters used in the dataset

**⚠️ Note:** You must place this file in the same directory as the script or modify the path in `load_data()`.

## 🚀 How to Run

1. Install required Python libraries:
   ```bash
   pip install torch numpy matplotlib tqdm
   ```

2. Make sure `deepwriting_training.npz` is in the same directory.

3. Run the script:
   ```bash
   python TASK5_hw_generation_using_RNN.py
   ```

The script will:
- Train the RNN model (with reduced epochs/sample size for testing)
- Generate handwriting strokes for sample text like `"a"` and `"b"`
- Visualize the output using `matplotlib`

## 🧠 Model Architecture

- **Embedding Layer** for characters
- **Bidirectional LSTM** (multi-layer, with dropout)
- Fully connected output layers producing `[dx, dy, pen]` stroke predictions
- **MSE Loss** used for training stroke predictions

## 📊 Training Results

- Training is limited to small samples (500) and epochs (2) for quick testing.
- Loss typically decreases steadily, and the model can generate visually coherent letter strokes even with minimal training.

## ✍️ Output

The output includes:
- Stroke-by-stroke plots simulating how the character(s) would be written by hand
- Can be extended to generate full sentences with longer training

## 📦 Output File

- Trained model weights saved as: `best_handwriting_model.pth`

## ✅ Future Improvements

- Use more text samples during generation
- Train on full dataset with more epochs
- Add GUI or interface for live handwriting generation
- Improve stroke rendering for smoother visuals
- -increase samples and epochs 

---
