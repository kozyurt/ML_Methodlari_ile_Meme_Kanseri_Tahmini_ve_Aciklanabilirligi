#Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Scikit-learn Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
# Değerlendirme Metriği Kütüphaneleri
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score, precision_score, recall_score
)
# Modelleme Libraries
import pickle
import os
# Explainable AI Libraries
import shap
#Modeller
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# Settings
pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.max_rows', None)     # Gerekirse tüm satırları göster
pd.set_option('display.width', None)        # Satırı bölmeden yaz
pd.set_option('display.max_colwidth', None) # Sütun isimlerini kesme

import warnings
warnings.filterwarnings('ignore')

# Dataset Upload
# If the "data.csv" file does not exist locally, retrieve the data from the internet.
try:
    df = pd.read_csv("data.csv")
except FileNotFoundError:
    print("File not found. Data is being retrieved from the internet...")
    df = pd.read_csv('https://www.kaggle.com/api/v1/datasets/download/uciml/breast-cancer-wisconsin-data/data.csv')
print("The dataset was successfully uploaded.")

# General Information About the Dataset
print("Data Set Size:", df.shape)
print("\nColumn Information:", df.info())
print(f"\nTarget Variable Distribution:{df['diagnosis'].value_counts()}")
print(f"\nTarget Variable Ratios:{df['diagnosis'].value_counts(normalize=True)*100}")

#Missing Value Check
print("\nMissing Value Check")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Target Variable Distribution and Ratios
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df['diagnosis'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Target Variable Distribution')
axes[0].set_xlabel('Diagnosis (M=Malignant, B=Benign)')
axes[0].set_ylabel = ('Count')
axes[0].set_xticklabels(['Benign', 'Malignant'], rotation=0)

df["diagnosis"].value_counts(normalize=True).plot(kind = 'pie', ax=axes[1],
                                                          autopct='%1.1f%%', colors=['green', 'red'],
                                                          labels = ['Benign', 'Malignant'])
axes[1].set_title('Target Variable Ratios')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig("graphs/diagnosis_distribution_eng.png", dpi=150, bbox_inches='tight')
plt.show()

# Correlation Matrix
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numerical_cols].corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr, cmap='coolwarm', center = 0, fmt = ".2f", annot=True, square=True, annot_kws={"size": 7})
plt.title('Correlation Matrix Between Features')
plt.savefig("graphs/correlation_matrix_eng.png", dpi=150, bbox_inches='tight')
plt.show()

df_clean = df.copy()
df_clean = df_clean.drop(columns=['id', 'Unnamed: 32']) # Removing unnecessary columns

encoder = LabelEncoder()
df_clean['diagnosis'] = encoder.fit_transform(df_clean['diagnosis']) # Converting the target variable to a numerical value (M=1, B=0)


X = df_clean.drop('diagnosis', axis=1)
y = df_clean['diagnosis']

print("Features Size:", X.shape)
print("Target Variable Size:", y.shape)

def count_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers)

outlier_counts = {}
for col in X.columns:
    outlier_counts[col] = count_outliers_iqr(df_clean, col)

outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['Number of Outliers'])
outlier_df = outlier_df.sort_values('Number of Outliers', ascending=False)
print(outlier_df.head(15))

top10_features = outlier_df.head(10).index.tolist()

plt.figure(figsize=(14, 6))

sns.boxplot(
    data=df_clean[top10_features],
)

plt.title("Boxplot for the 10 Feature with the Most Outliers (IQR)", fontsize=14)
plt.xlabel("Value")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("graphs/top10_outliers_boxplot_eng.png", dpi=150, bbox_inches='tight')
plt.show()


# Feature Engineering Transformer (foe Pipeline)
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_:
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                X = pd.DataFrame(X)

        X = X.copy()

        if 'radius_mean' in X.columns and 'area_mean' in X.columns:
            X['radius_area_ratio'] = X['radius_mean'] / (X['area_mean'] + 1e-6)

        if 'perimeter_mean' in X.columns and 'area_mean' in X.columns:
            X['perimeter_area_ratio'] = X['perimeter_mean'] / (X['area_mean'] + 1e-6)

        if 'concavity_mean' in X.columns and 'concave points_mean' in X.columns:
            X['concavity_points_product'] = X['concavity_mean'] * X['concave points_mean']

        return X


print("FeatureEngineeringTransformer is done.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train Features Dimension:", X_train.shape)
print("Test Features Dimension:", X_test.shape)
print("Train Target Variable Dimension:", y_train.shape)
print("Test Target Variable Dimension:", y_test.shape)

# Preprocessing Pipeline Creation
preprocessing_pipeline = Pipeline([
    ('feature_engineering', FeatureEngineeringTransformer()),
    ('scaler', StandardScaler())
])
X_train_scaled = preprocessing_pipeline.fit_transform(X_train)
X_test_scaled = preprocessing_pipeline.transform(X_test)

# Returned as DataFrame
feature_eng = FeatureEngineeringTransformer().fit_transform(X_train)
feature_names = feature_eng.columns.tolist()

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

print(f"Preprocessing with the pipeline is complete.")
print(f"Train set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")
print(f"Derived Features: {[col for col in X_train_scaled.columns if col not in X_train.columns]}")
print("Now all preprocessing is done through the pipeline!")

# Correlation between Objective Variable and Features (Preprocessed with Pipeline)
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X_train_scaled, y_train)

feature_importance = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
plt.title('Correlation Between Target Variable and Features (Preprocessed with Pipeline)')
plt.xlabel('Correlation Value')
plt.ylabel('Features')
plt.grid()
plt.tight_layout()
plt.savefig("graphs/target_correlation_eng.png", dpi=150, bbox_inches='tight')
plt.show()
print("Top 10 Correlative Features:")
print(feature_importance.head(10))


#Model Selection
results = {}

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=10000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    "XGBoost": xgb.XGBClassifier(n_estimatoers=200, learning_rate=0.1, max_depth = 5,
                                 random_state=42, evaluation_metric='logloss', use_label_encoder=False),
    "LigthtGBM": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)
}

print("The models are being trained...")
for model_name, model in models.items():
    print(f"{model_name} model is being trained...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results[model_name] = {
        'model_name': model,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV_Mean': cv_mean,
        'CV_Std': cv_std,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, CV Mean: {cv_mean:.4f}")

model_comparison_df = pd.DataFrame({
    "Model" : list(results.keys()),
    "Accuracy": [results[model]['Accuracy'] for model in results.keys()],
    "Precision": [results[model]['Precision'] for model in results.keys()],
    "Recall": [results[model]['Recall'] for model in results.keys()],
    "F1-Score": [results[model]['F1-Score'] for model in results.keys()],
    "ROC-AUC": [results[model]['ROC-AUC'] for model in results.keys()],
    "CV_Mean": [results[model]['CV_Mean'] for model in results.keys()],
    "CV_Std": [results[model]['CV_Std'] for model in results.keys()]
}).sort_values(by='F1-Score', ascending=False)

print("\n" + model_comparison_df.to_string(index=False))

best_model_name = model_comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model_name']
print(f"\nBest Model: {best_model_name}")
print(f"F1-Score: {results[best_model_name]['F1-Score']:.4f}")
print(f"Accuracy: {results[best_model_name]['Accuracy']:.4f}")
print(f"ROC-AUC: {results[best_model_name]['ROC-AUC']:.4f}")

#Detailed Analysis of the Best Model (Logistic Regression)
fig, axes = plt.subplots(2, 2, figsize=(25, 15))

ax1 = axes[0, 0]
model_comparison_df.plot(x='Model', y='Accuracy', kind='barh', ax=ax1, color='skyblue', legend=False)
ax1.axvline(x=results[best_model_name]['Accuracy'], color='r', linestyle='--')
ax1.set_title('Model Comparison - Accuracy')
ax1.set_xlabel('Accuracy')
ax1.set_xlim([0.8, 1.0])

ax2 = axes[0, 1]
metrics_comparison = ["Accuracy", "Precision", "Recall", "F1-Score"]
model_comparison_df.plot(x='Model', y=metrics_comparison, kind='barh', ax=ax2)
ax2.set_title('Model Comparison - Key Metrics')
ax2.set_xlabel('Value')
ax2.set_xlim([0.8, 1.0])

ax3 = axes[1, 0]
model_comparison_df.plot(x='Model', y='ROC-AUC', kind='barh', ax=ax3, color='lightgreen', legend=False)
ax3.axvline(x=results[best_model_name]['ROC-AUC'], color='r', linestyle='--')
ax3.set_title('Model Comparison - ROC-AUC')
ax3.set_xlabel('ROC-AUC')
ax3.set_xlim([0.8, 1.0])

ax4 = axes[1, 1]
ax4.barh(model_comparison_df['Model'], model_comparison_df['CV_Mean'], xerr=model_comparison_df['CV_Std'],
         color='lightgreen', edgecolor='darkgreen', capsize=5)
ax4.axvline(x=results[best_model_name]['CV_Mean'], color='r', linestyle='--')
ax4.set_title('Model Comparison - Cross-Validation (CV) Mean and Std')
ax4.set_xlabel('CV Mean')
ax4.set_xlim([0.8, 1.0])

plt.savefig('graphs/model_comparison_eng.png', dpi=300, bbox_inches='tight')

#Focusing on Logistic Regression
print(f"{best_model_name} Modeli Detaylı İnceleme:")

y_pred = results[best_model_name]['y_pred']
y_proba = results[best_model_name]['y_proba']

print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malignant (1)']))
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(3, 1, figsize=(10, 18))

ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.xaxis.set_ticklabels(['Benign (0)', 'Malignant (1)'])
ax1.yaxis.set_ticklabels(['Benign (0)', 'Malignant (1)'])

ax2 = axes[1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
ax2.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {results[best_model_name]["ROC-AUC"]:.4f})')
ax2.plot([0, 1], [0, 1], color='red', linestyle='--')
ax2.set_title('ROC Curve')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend()
ax2.grid()

ax3 = axes[2]
ax3.hist(y_proba[y_test == 0], bins=25, alpha=0.6, label='Benign (0)', color='green')
ax3.hist(y_proba[y_test == 1], bins=25, alpha=0.6, label='Malignant (1)', color='red')
ax3.set_title('Estimation Probability Distribution')
ax3.set_xlabel('Predicted Probability')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid()

plt.savefig('graphs/best_model_detailed_analysis_eng.png', dpi=300, bbox_inches='tight')

plt.show()

param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'num_leaves': [31, 50, 70]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'poly']
    },
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }
}

if best_model_name in param_grids:
    param_grid = param_grids[best_model_name]

    print(f"Hyperparameter adjustment is being performed for the best model.")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")

    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    print(f"\nBest parameters:")
    for key, value in grid_search.best_params_.items():
        print(f"  {key}: {value}")

    print(f"The Best CV Score: {grid_search.best_score_:.4f}")

    # Evaluate tuned model
    tuned_model = grid_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test_scaled)
    y_pred_proba_tuned = tuned_model.predict_proba(X_test_scaled)[:, 1]

    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    f1_tuned = f1_score(y_test, y_pred_tuned)
    roc_auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)

    print(f"Tuned Model Performance:")
    print(f"  Accuracy: {accuracy_tuned:.4f}")
    print(f"  F1-Score: {f1_tuned:.4f}")
    print(f"  ROC-AUC: {roc_auc_tuned:.4f}")

    print(f"Performance improvements after hyperparameter adjustment:")
    print(f"  Accuracy: {(accuracy_tuned - results[best_model_name]['Accuracy']) * 100:+.2f}%")
    print(f"  F1-Score: {(f1_tuned - results[best_model_name]['F1-Score']) * 100:+.2f}%")
    print(f"  ROC-AUC: {(roc_auc_tuned - results[best_model_name]['ROC-AUC']) * 100:+.2f}%")

    best_model = tuned_model

else:
    print(f"GridSearchCV is undefined for {best_model_name}")
    print("The original model is being used...")
    tuned_model = best_model

#Using Ensemble Models
top_models = model_comparison_df.head(3)['Model'].tolist()
print("Top 3 Model:", top_models)

estimators = [(name, results[name]['model_name']) for name in top_models]
voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
voting_clf.fit(X_train_scaled, y_train)

y_pred_ensamble = voting_clf.predict(X_test_scaled)
y_proba_ensamble = voting_clf.predict_proba(X_test_scaled)[:, 1]

accuracy_ensamble = accuracy_score(y_test, y_pred_ensamble)
precision_ensamble = precision_score(y_test, y_pred_ensamble)
recall_ensamble = recall_score(y_test, y_pred_ensamble)
f1_ensamble = f1_score(y_test, y_pred_ensamble)
roc_auc_ensamble = roc_auc_score(y_test, y_proba_ensamble)

print("\nEnsemble Model Performance:")
print(f"Accuracy: {accuracy_ensamble:.4f}, Precision: {precision_ensamble:.4f}, Recall: {recall_ensamble:.4f}, F1-Score: {f1_ensamble:.4f}, ROC-AUC: {roc_auc_ensamble:.4f}")

print(f"Comparison of the best model we have, {best_model_name}, with the Ensemble Model.")
print(f"Accuracy: {(accuracy_ensamble - results[best_model_name]['Accuracy'])*100:+.2f}%")
print(f"F1-Score: {(f1_ensamble - results[best_model_name]['F1-Score'])*100:+.2f}%")
print(f"ROC-AUC: {(roc_auc_ensamble - results[best_model_name]['ROC-AUC'])*100:+.2f}%")

#Last Comparing
if f1_ensamble > f1_tuned:
    print(f"The Ensemble model performed better than the best model.")
    final_model = voting_clf
    final_model_name = "Ensemble Model (VotingClassifier)"
    y_pred_final = y_pred_ensamble
    y_proba_final = y_proba_ensamble
else:
    print(f"The {best_model_name} model showed the best performance.")
    final_model_name = best_model_name
    final_model = tuned_model
    y_pred_final = y_pred_tuned
    y_proba_final = y_pred_proba_tuned


print(f"SHAP Analysis: {best_model_name} Model")
explainer = shap.LinearExplainer(best_model, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

print(f"\nSHAP values were calculated. Shape: {np.array(shap_values).shape}")

# SHAP Summary Plot (Bar) - Feature priority ranking
plt.figure(figsize=(18, 15))
shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False, max_display=15)
plt.title(f'SHAP Feature Importance - {best_model_name}')
plt.tight_layout()
plt.savefig("graphs/shap_importance_eng.png", dpi=150, bbox_inches='tight')
plt.show()

# SHAP Summary Plot (Dot) - Detailed feature effect
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_scaled, show=False, max_display=15)
plt.title(f'SHAP Feature Impact Analysis - {best_model_name}')
plt.tight_layout()
plt.savefig("graphs/shap_summary_eng.png", dpi=150, bbox_inches='tight')
plt.show()

#Final Model Summary
print("Final Model: ", final_model_name)

final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_roc_auc = roc_auc_score(y_test, y_proba_final)

print(f"Final Model Performance:")
print(f"  Accuracy: {final_accuracy:.4f}")
print(f"  Precision: {final_precision:.4f}")
print(f"  Recall: {final_recall:.4f}")
print(f"  F1-Score: {final_f1:.4f}")
print(f"  ROC-AUC: {final_roc_auc:.4f}")

cm_final = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues')
plt.title(f'Final Model Confusion Matrix - {final_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Benign (0)', 'Malignant (1)'])
plt.tight_layout()
plt.savefig("graphs/final_model_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()


# Pipeline: Feature Engineering → StandardScaler → Final Model
final_pipeline = Pipeline([
    ('feature_engineering', FeatureEngineeringTransformer()),
    ('scaler', StandardScaler()),
    ('model', final_model)  # We are using your current final_model.
])

# Train the pipeline (using X_train, preprocessing will be done automatically)
print(f"Pipeline is training (Model: {final_model_name})...")
final_pipeline.fit(X_train, y_train)

# Make predictions using pipelines.
y_pred_pipeline = final_pipeline.predict(X_test)
y_proba_pipeline = final_pipeline.predict_proba(X_test)[:, 1]

# Check the performance
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)
f1_pipeline = f1_score(y_test, y_pred_pipeline)
roc_auc_pipeline = roc_auc_score(y_test, y_proba_pipeline)

print(f"\nPipeline Performance:")
print(f"  Accuracy: {accuracy_pipeline:.4f}")
print(f"  F1-Score: {f1_pipeline:.4f}")
print(f"  ROC-AUC: {roc_auc_pipeline:.4f}")


print("MODEL SAVING")

# Save the pipeline
model_filename = 'breast_cancer_pipeline.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(final_pipeline, f)

print(f"Pipeline successfully saved: {model_filename}")

# Also save model information (optional)
model_info = {
    'model_name': final_model_name,
    'accuracy': accuracy_pipeline,
    'f1_score': f1_pipeline,
    'roc_auc': roc_auc_pipeline,
    'feature_names': list(X_train.columns),
    'created_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}
info_filename = 'model_info.pkl'
with open(info_filename, 'wb') as f:
    pickle.dump(model_info, f)

print(f"Model information has been recorded: {info_filename}")

print("MODEL UPLOADING")

# Upload the model (for testing)
loaded_pipeline = pickle.load(open(model_filename, 'rb'))
loaded_info = pickle.load(open(info_filename, 'rb'))

print(f"Pipeline is uploaded: {loaded_info['model_name']}")
print(f"Model performance: Accuracy={loaded_info['accuracy']:.4f}, F1={loaded_info['f1_score']:.4f}")

print("\nModel saving/loading process is complete!")

print("YENİ VERİ İLE TAHMİN YAPMA")

# Örnek: Test setinden birkaç örnek al
print("\nÖrnek: Test setinden 5 örnek alınıyor...\n")
yeni_veri = X_test.head(5).copy()


# Pipeline ile tahmin yap (Preprocessing otomatik yapılacak!)
print("PIPELINE İLE TAHMİN YAPILIYOR...")

#Kaydedilmiş pipeline'ı yükle ve kullan
loaded_pipeline = pickle.load(open('breast_cancer_pipeline.pkl', 'rb'))
tahminler = loaded_pipeline.predict(yeni_veri)
olasiliklar = loaded_pipeline.predict_proba(yeni_veri)[:, 1]

# Show the results
sonuclar = pd.DataFrame({
    'Prediction': tahminler,
    'Probability (Malignant)': olasiliklar,
    'Actual Values': y_test.iloc[:5].values,
    'Is it True?': (tahminler == y_test.iloc[:5].values)
})

print("\nPrediction Results:")
print(sonuclar.to_string(index=False))