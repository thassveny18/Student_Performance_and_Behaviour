# Student_Performance_and_Behaviour (Group 2)

A complete end-to-end machine learning pipeline for predicting student academic grades using Random Forest and XGBoost classifiers.

---

## Dataset

**Source:** [Students Grading Dataset — Kaggle](https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset/data)

The dataset contains student academic performance features including attendance, study habits, scores, department, gender, and total score.

**Target Variable:** `Grade`

| Grade | Encoded Value |
|-------|---------------|
| A     | 4             |
| B     | 3             |
| C     | 2             |
| D     | 1             |
| F     | 0             |

---

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn

---

## Installation

Install all required dependencies:

```bash
pip install xgboost scikit-learn pandas matplotlib seaborn imbalanced-learn
```
> **Note:** If running on Google Colab, `imbalanced-learn` is installed as a separate cell mid-notebook before the SMOTETomek step.
 
---

## How to Run

The full pipeline is contained in a single .py file. Simply run it top to bottom:

```
ml_5.py
```
> **Note:** The notebook runs on Google Colab. When prompted, upload `Students Performance Dataset.csv` via the `files.upload()` cell. You will also need to mount your Google Drive to save generated plots automatically to `student_grade_prediction/images/`.

This notebook covers the complete workflow:

1. **Data Preprocessing** — duplicate removal, missing value handling, outlier removal (IQR), feature encoding, normalisation, and train/test split
2. **Exploratory Data Analysis** — correlation heatmap, grade distribution, attendance vs. score scatter plot, outlier boxplots
3. **Model Training** — Random Forest and XGBoost classifiers
4. **Hyperparameter Tuning** — `GridSearchCV` for both models; early stopping for XGBoost
5. **Class Imbalance Handling** — SMOTETomek resampling, `class_weight='balanced'` for Random Forest, sample weights for XGBoost
6. **Final Evaluation** — accuracy, precision, recall, F1-score, confusion matrices, and error analysis

---

## Project Structure

```
project/
│
├── README.md
├── Students Performance Dataset.csv
├── ml_1.py
├── ml_2.py
├── ml_3.py
├── ml_4.py
├── ml_5.py

```

---

## Results Summary

| Model         | Strength                                         |
|---------------|--------------------------------------------------|
| Random Forest | Stable and interpretable                         |
| XGBoost       | Higher predictive power and better optimisation  |

**Best-performing model:** XGBoost

**Key findings:**
- Both models achieved strong predictive performance
- Hyperparameter tuning successfully reduced overfitting
- SMOTETomek improved minority class prediction
- Most prediction errors were near-boundary misclassifications (e.g., predicting B instead of A)

---

## Author

- THASSVENY A/P SURIA KUMARAN
- MANIEMALAR A/P MONEYUAL
- ALLEN PAUL JUSTINUS SITU
- VIMALRAJ A/L SELVARAJU
- CHEONG YU JING

---

## License

This project is for educational and academic purposes only.
