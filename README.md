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
