# ASD_Classification
Integrative Gene Expression Analysis for Autism Biomarker Identification and Classification Using Machine Learning

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition characterized by deficits in social interaction, communication, and behavior. This study aims to identify genetic biomarkers for ASD by analyzing gene expression data and developing predictive models for classifying ASD and control samples. Using dataset GDS4431 from the Gene Expression Omnibus (GEO) database, differential expression analysis was conducted with t-tests and adjusted p-values using the Benjamini-Hochberg method. The identified significant DEGs were refined using Recursive Feature Elimination (RFE) with various classifiers to determine the optimal features for the models, enhancing their predictive accuracy. Hyperparameter tuning via Optuna optimized model performance, and all individual models and model combinations were evaluated using stratified k-fold cross-validation, with ROC-AUC as the primary metric. SHapley Additive exPlanation (SHAP) values provided insights into feature importance for models. This integrative approach identified key genetic markers and developed robust models for ASD classification, offering a foundation for future research and potential clinical applications.

Input Data: https://drive.google.com/drive/folders/1VeY7u-8mL0aIpB1iraiO0QPepPWCMwRJ?usp=drive_link

Gene Expression Analysis for ASD Classification and Biomarker Discovery
Project Overview
This project aims to analyze gene expression data for Autism Spectrum Disorder (ASD) classification and biomarker discovery. Using a combination of machine learning models and statistical techniques, the project identifies significantly differentially expressed genes and leverages various algorithms for ASD classification. The goal is to determine genetic biomarkers that can reliably distinguish between autism and control samples.

The analysis uses multiple models, including Random Forest, Gradient Boosting, XGBoost, CatBoost, SVM, and MLP (Multi-layer Perceptron), and integrates advanced techniques like Recursive Feature Elimination (RFE) and SHAP values to understand feature importance and model interpretability. Hyperparameter optimization is conducted using Optuna for enhanced model performance.

Key Features
Data Preprocessing:

Missing values are handled using KNN imputation, and genes with over 20% missing values are removed.
Gene expression data is normalized using log2 transformation.
Gene filtering is applied to remove those with no variance across samples.
T-tests and Multiple Testing Correction:

Performs t-tests between autistic and control samples to identify significantly differentially expressed genes.
Adjusts p-values for multiple comparisons using the Benjamini-Hochberg procedure.
Recursive Feature Elimination (RFE):

Uses RFE to identify the optimal number of features for classification models, selecting the most important genes that contribute to the classification task.
Model Training and Evaluation:

Trains various machine learning models (Random Forest, Gradient Boosting, XGBoost, CatBoost, SVM, and MLP) on the identified genes.
Evaluates models based on accuracy, ROC-AUC score, and classification reports.
Hyperparameter Optimization:

Uses Optuna to optimize the hyperparameters of the models, improving their performance based on ROC-AUC scores.
Model Combination:

Evaluates combinations of models (ensembles) to determine whether combining models leads to better performance than individual models.
SHAP Values:

Generates SHAP values to explain model predictions and identify the most important genes contributing to ASD classification.
Model Visualization:

Visualizes model performance (accuracy, ROC-AUC, precision, recall, F1 score) and presents SHAP values to explain feature importance.
Plots PCA (Principal Component Analysis) to visualize test data colored by predicted probabilities from the best-performing model.
Requirements
To run this project, you need the following Python libraries:

Pandas and NumPy for data manipulation:

bash
Copy code
pip install pandas numpy
Scikit-learn for machine learning models, preprocessing, and evaluation:

bash
Copy code
pip install scikit-learn
Seaborn and Matplotlib for visualization:

bash
Copy code
pip install seaborn matplotlib
Imbalanced-learn for SMOTE (handling imbalanced data):

bash
Copy code
pip install imbalanced-learn
XGBoost, CatBoost, and Optuna for additional models and hyperparameter optimization:

bash
Copy code
pip install xgboost catboost optuna
SHAP for model interpretability:

bash
Copy code
pip install shap
Logging for tracking model runs and results:

Already part of Python's standard library.
Project Structure
Key Classes and Functions
Data Preprocessing:

Handles missing values, normalizes gene expression data, and filters out genes with no variance.
Splits data into training and test sets and applies SMOTE for handling class imbalance.
T-tests and Multiple Testing Correction:

Performs t-tests for each gene, comparing the autism group with the control group.
Adjusts p-values using FDR (False Discovery Rate) correction and selects significant genes for further analysis.
Recursive Feature Elimination (RFE):

Uses RFE to iteratively eliminate less important genes and determine the optimal number of features for classification models.
Model Training and Hyperparameter Optimization:

Trains models using stratified k-fold cross-validation and evaluates their ROC-AUC scores.
Hyperparameters are optimized using Optuna, which searches for the best hyperparameters based on ROC-AUC scores.
Model Combination and Ensemble Evaluation:

Evaluates model combinations to explore whether ensembles perform better than individual models.
SHAP Values:

Generates SHAP values to explain the impact of each gene on model predictions and visualize the importance of the selected features.
Logging and Visualization:

Logs model parameters and results to a file.
Visualizes performance metrics (accuracy, ROC-AUC, precision, recall, F1 score) using Matplotlib and Seaborn.
Model Training and Evaluation Process
Train Models:

Models are trained using the selected genes after RFE.
The models include Gradient Boosting, XGBoost, SVM, and MLP.
Evaluate Models:

Each model is evaluated using metrics like accuracy, ROC-AUC, precision, recall, and F1 score.
The best-performing model is determined based on ROC-AUC.
SHAP and PCA Visualization:

SHAP values are generated for feature importance interpretation.
PCA is used to visualize the test data based on predicted probabilities from the best-performing model.
Usage
Steps to Run the Project:
Prepare the Data:

Ensure that you have the gene expression dataset (GDS4431.csv) and the corresponding disease state labels (Disease_State.csv).
Run the Script:

Load the data, preprocess it, perform t-tests, and filter significant genes.
Run RFE to select the optimal number of features.
Train the models and evaluate their performance using cross-validation.
python
Copy code
# Load and preprocess the data
disease_state_df = pd.read_csv(disease_state_path).set_index('samples')
gds_df = pd.read_csv(gds_path).set_index('ID_REF')

# Process missing values, normalize the data, and filter significant genes
# Split data and apply SMOTE for balancing
X_train, X_test, y_train, y_test = train_test_split(X_rfe, y_rfe, test_size=0.3, random_state=42)

# Train and evaluate the models
for model_name, model in final_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Model: {model_name}, ROC-AUC: {roc_auc:.4f}")

# Plot SHAP values
for model_name, model in final_models.items():
    plot_shap_feature_importance(model, model_name, X_test, optimal_rfe_genes['IDENTIFIER'].values)
Example Output:
Model Evaluation:

yaml
Copy code
Model: GradientBoosting, ROC-AUC: 0.93
Model: XGBoost, ROC-AUC: 0.91
Model: SVM, ROC-AUC: 0.88
Model: MLP, ROC-AUC: 0.90
Best Individual Model: GradientBoosting with ROC-AUC of 0.93.

Best Model Combination: An ensemble of models with ROC-AUC of 0.94.

SHAP Values Visualization: A bar chart showing the most important genes for each model.

Conclusion
This project provides a comprehensive analysis of gene expression data for ASD classification and biomarker discovery. Using statistical tests, feature selection techniques like RFE, and advanced machine learning models, it identifies key genes associated with autism. The models are optimized using Optuna for hyperparameter tuning, and SHAP values are used to interpret the predictions, making the analysis interpretable and reliable.

Future Enhancements
Additional Models: Explore other machine learning models, such as LightGBM or ElasticNet, to improve classification performance.
Advanced Feature Selection: Implement feature selection techniques such as LASSO or ElasticNet for comparison with RFE.
Integration with External Datasets: Combine this analysis with external autism datasets to validate the findings and improve model generalization.
