# Parkinson's Disease Detection using Machine Learning

This project uses biomedical voice measurements to classify whether a patient has Parkinson's disease. Several classification algorithms are trained and compared to identify the best-performing model.

## Dataset

- **Source:** [Parkinson's Disease Data Set](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set) (`parkinsons.data`)
- **Description:** The dataset contains a range of biomedical voice measurements from individuals, some of whom have Parkinson's disease (PD). Each row corresponds to one voice recording, and each column is a particular voice measure.
- **Target column:** `status` — 1 indicates the presence of Parkinson's disease, 0 indicates a healthy individual.
- The `name` column (patient/recording identifier) is dropped before modeling since it carries no predictive value.

## Project Workflow

1. **Data Loading & Cleaning**
   - Load the dataset with `pandas`.
   - Drop the non-predictive `name` column.
   - Inspect data types and check for missing values.

2. **Exploratory Data Analysis (EDA)**
   - Visualize the class distribution of the `status` target using a bar plot.
   - Plot distribution plots (`displot`) for every feature to understand their spread and skewness.
   - Generate a correlation heatmap to examine relationships between features.

3. **Handling Class Imbalance**
   - The dataset is imbalanced (more PD-positive cases than healthy ones).
   - `RandomOverSampler` from the `imbalanced-learn` library is used to oversample the minority class and balance the dataset.

4. **Feature Scaling & Dimensionality Reduction**
   - Features are scaled to the range **[-1, 1]** using `MinMaxScaler`.
   - **PCA** (Principal Component Analysis) is applied, retaining enough components to explain 95% of the variance, to reduce dimensionality while preserving information.

5. **Train/Test Split**
   - Data is split into training and testing sets (80/20 split, `random_state=7`).

6. **Model Training & Evaluation**
   The following classification models are trained on the PCA-transformed data and evaluated using accuracy score:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)

7. **Model Comparison**
   - Accuracy scores for all models are compiled into a summary table, sorted in descending order.
   - A bar chart visualizes and compares the accuracy of each model.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

## How to Run

1. Place `parkinsons.data` in the expected input directory (e.g. `/kaggle/input/parkinsons-disease-data-set/parkinsons.data` if running on Kaggle, or update the path if running locally).
2. Run the notebook cells in order:
   - Data loading and preprocessing
   - EDA and visualization
   - Oversampling, scaling, and PCA
   - Model training and evaluation
   - Model comparison

## Results

The notebook outputs a comparison table and bar chart of accuracy scores for Logistic Regression, Decision Tree, Random Forest, SVM, and KNN classifiers, making it easy to identify the best-performing model for this dataset.

## Notes / Possible Improvements

- Only accuracy is currently compared across models; consider also reporting precision, recall, F1-score, and confusion matrices (the relevant imports are already included) for a more complete evaluation, especially important for medical diagnosis tasks.
- Hyperparameter tuning (e.g. `GridSearchCV`) could further improve model performance.
- Cross-validation would give a more robust estimate of model performance than a single train/test split.
