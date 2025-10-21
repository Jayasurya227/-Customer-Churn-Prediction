# Customer Churn Prediction 📊

This project focuses on predicting customer churn for a telecom company using demographic, account, and service usage data. The goal is to build classification models that can identify customers likely to leave (churn), enabling proactive retention strategies.

The notebook demonstrates a complete machine learning workflow for a binary classification problem, including data cleaning, exploratory data analysis (EDA), feature engineering/transformation, preprocessing with Scikit-learn pipelines, training multiple models (Logistic Regression and Random Forest), and comprehensive evaluation using various metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, AUC-ROC).

**Dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Telco Customer Churn dataset)
**Target Variable:** `Churn` (Yes/No, converted to 1/0)
**Focus:** Demonstrating data cleaning, EDA for classification, feature transformation (log transform), preprocessing pipelines, model comparison (Logistic Regression vs. Random Forest), evaluation metrics for classification, and feature importance analysis.
**Repository:** [https://github.com/Jayasurya227/-Customer-Churn-Prediction](https://github.com/Jayasurya227/-Customer-Churn-Prediction)

***

## Key Techniques & Concepts Demonstrated

Based on the analysis within the notebook (`7_Preventing_Customer_Churn_with_Feature_Transformation.ipynb`), the following key concepts and techniques are applied:

* **Binary Classification:** Predicting one of two outcomes (Churn vs. No Churn).
* **Data Cleaning:**
    * Identifying and converting the `TotalCharges` column from object to numeric type after handling inconsistent entries (empty strings).
    * Imputing missing `TotalCharges` values (created during conversion) using the median.
    * Dropping irrelevant identifiers (`customerID`).
* **Exploratory Data Analysis (EDA):**
    * Analyzing the distribution of the target variable (`Churn`), revealing class imbalance.
    * Visualizing relationships between categorical features (e.g., `Contract`, `InternetService`, `PaymentMethod`) and churn using count plots.
    * Visualizing distributions of numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) split by churn status using histograms/KDE plots.
    * Identifying and addressing skewness in `TotalCharges` using a **log transformation** (`np.log1p`).
    * Examining correlations between numerical features using a heatmap.
* **Feature Engineering & Transformation:**
    * Applying log transformation to the skewed `TotalCharges` feature.
    * Mapping the target variable 'Churn' from Yes/No to 1/0.
* **Data Preprocessing within Pipelines:**
    * **Imputation:** Using `SimpleImputer` (median for numerical, most frequent for categorical).
    * **Feature Scaling:** Applying `StandardScaler` to numerical features.
    * **Categorical Encoding:** Using `OneHotEncoder` (with `drop='first'`) for categorical features.
    * **Pipeline & ColumnTransformer:** Creating a unified preprocessor for systematic handling of different feature types.
* **Model Building & Comparison:**
    * Training a baseline **Logistic Regression** model.
    * Training an ensemble **Random Forest Classifier** model.
    * Both models integrated into Scikit-learn Pipelines with the preprocessing steps.
* **Model Evaluation:**
    * **Accuracy Score:** Overall prediction correctness.
    * **Classification Report:** Detailed precision, recall, and F1-score for each class (Churn/No Churn).
    * **Confusion Matrix:** Visualizing True Positives, False Positives, True Negatives, and False Negatives.
    * **AUC-ROC Curve:** Plotting the Receiver Operating Characteristic curve and calculating the Area Under the Curve (AUC) to evaluate the model's ability to distinguish between classes across different thresholds.
* **Feature Importance:** Extracting and visualizing feature importances from the Random Forest model to understand key drivers of churn.

***

## Key Findings & Insights

* **EDA Highlights:** Features like `Contract` duration (Month-to-month contracts have higher churn), `tenure` (lower tenure correlates with higher churn), `InternetService` (Fiber optic customers churn more), and `PaymentMethod` (Electronic check users churn more) show strong relationships with churn. `TotalCharges` was found to be right-skewed and benefited from a log transformation.
* **Model Performance:** The Random Forest Classifier significantly outperformed Logistic Regression, achieving higher accuracy, better precision/recall balance (F1-score), and a higher AUC score, indicating better overall discriminative power.
* **Feature Importance:** The Random Forest model identified `Contract` type, `tenure`, `TotalCharges` (log-transformed), `MonthlyCharges`, and `PaymentMethod` (Electronic Check) as the most important predictors of churn.

***

## Analysis Workflow

The notebook follows a standard classification workflow:

1.  **Setup & Data Loading:** Importing libraries and loading the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset.
2.  **Initial Data Exploration & Cleaning:**
    * Inspecting data types, summary statistics, and missing values.
    * Handling inconsistencies and converting `TotalCharges` to numeric, followed by median imputation.
    * Dropping the `customerID` column.
3.  **Exploratory Data Analysis (EDA):**
    * Visualizing the target variable distribution.
    * Plotting categorical and numerical features against the 'Churn' target.
    * Checking numerical feature distributions and applying log transformation to `TotalCharges`.
    * Plotting a correlation heatmap.
4.  **Preprocessing & Feature Transformation:**
    * Mapping 'Churn' to binary (1/0).
    * Defining numerical and categorical feature lists.
    * Creating preprocessing pipelines using `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.
    * Combining pipelines with `ColumnTransformer`.
    * Splitting data into stratified train and test sets.
5.  **Model Training:**
    * Building full pipelines including the preprocessor and classifier for Logistic Regression and Random Forest.
    * Fitting both pipelines on the training data.
6.  **Model Evaluation:**
    * Generating predictions on the test set.
    * Calculating and printing accuracy, classification reports, and confusion matrices.
    * Calculating AUC scores and plotting ROC curves for comparison.
7.  **Feature Importance Analysis:**
    * Extracting feature names post-encoding from the Random Forest pipeline's preprocessor.
    * Getting feature importances from the trained Random Forest classifier.
    * Plotting the top N important features.
8.  **Conclusion:** Summarizing the process, model performance comparison, and key insights.

***

## Technologies Used

* **Python**
* **Pandas & NumPy:** For data loading, manipulation, cleaning, and transformation.
* **Matplotlib & Seaborn:** For data visualization.
* **Scikit-learn:** For data splitting (`train_test_split`), preprocessing (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`), modeling (`LogisticRegression`, `RandomForestClassifier`), and evaluation metrics (`accuracy_score`, `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve`).
* **Jupyter Notebook / Google Colab:** For the interactive analysis environment.

***

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jayasurya227/-Customer-Churn-Prediction.git](https://github.com/Jayasurya227/-Customer-Churn-Prediction.git)
    cd -Customer-Churn-Prediction 
    ```
    *(Note: You might want to rename the repository later to remove the leading hyphen for easier command-line access)*
2.  **Install dependencies:**
    (It is recommended to use a virtual environment)
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
3.  **Ensure Dataset:** Make sure the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file is present in the repository directory.
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook "7_Preventing_Customer_Churn_with_Feature_Transformation.ipynb"
    ```
    *(Run the cells sequentially to perform the analysis.)*

***

## Author & Portfolio Use

* **Author:** Jayasurya227
* **Portfolio:** This project ([https://github.com/Jayasurya227/-Customer-Churn-Prediction](https://github.com/Jayasurya227/-Customer-Churn-Prediction)) provides a solid example of solving a common business problem (customer churn) using binary classification. It demonstrates skills in data cleaning, EDA, feature transformation, using Scikit-learn pipelines, model comparison, and interpreting results. Suitable for showcasing on GitHub, resumes/CVs, LinkedIn, and during interviews for data analyst or data scientist roles.
* **Notes:** Recruiters can assess the structured approach to problem-solving, data handling, model building, evaluation using multiple relevant metrics (especially for imbalanced data), and the ability to extract actionable insights via feature importance.
