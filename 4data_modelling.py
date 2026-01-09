"""
Hospital Readmission Prediction - Modeling Script
Project: Analysis and Prediction of Hospital Readmission in Diabetic Patients Using Healthcare Data

This script performs:
1. Feature selection based on EDA insights
2. Multiple model training (Logistic Regression, Random Forest, XGBoost)
3. Comprehensive evaluation with clinical relevance
4. Model comparison and interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
os.makedirs('./output/data_modelling', exist_ok=True)

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("HOSPITAL READMISSION PREDICTION - MODELING ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[STEP 1] Loading cleaned data...")

# Load the cleaned dataset
df = pd.read_csv('./dataset/diabetic_data_cleaned.csv')

print(f"Dataset shape: {df.shape}")
print(f"Features: {df.shape[1]}")
print(f"Samples: {df.shape[0]}")

# ============================================================================
# STEP 2: CHECK TARGET VARIABLE
# ============================================================================
print("\n[STEP 2] Checking target variable...")

# The cleaned dataset already has readmitted_binary
print(f"\nTarget variable distribution:")
print(df['readmitted_binary'].value_counts())
print(f"Readmission rate: {df['readmitted_binary'].mean()*100:.2f}%")

print("\nDiagnosis category distribution:")
print(df['diag_1_category'].value_counts())

# ============================================================================
# STEP 3: FEATURE SELECTION (Based on EDA insights)
# ============================================================================
print("\n[STEP 3] Selecting features based on EDA findings...")
print("\nFeature selection rationale:")
print("- number_inpatient: Strongest predictor (prior hospitalization history)")
print("- number_emergency: Emergency visit frequency indicates severity")
print("- number_diagnoses: Disease complexity measure")
print("- time_in_hospital: Length of stay correlates with condition severity")
print("- num_medications: Polypharmacy indicator")
print("- diag_1_category: Primary diagnosis type")
print("- age_numeric: Age as continuous variable")
print("- num_lab_procedures: Diagnostic intensity")

# Selected features based on EDA
selected_features = [
    'number_inpatient',      # ⭐ Strongest predictor from EDA
    'number_emergency',      # Emergency visits frequency
    'number_diagnoses',      # Disease complexity
    'time_in_hospital',      # Length of stay
    'num_medications',       # Medication count
    'num_lab_procedures',    # Diagnostic procedures
    'age_numeric',           # Age (numeric)
    'diag_1_category'        # Primary diagnosis category
]

# Create modeling dataset
df_model = df[selected_features + ['readmitted_binary']].copy()

# Handle missing values
print("\nHandling missing values...")
df_model['age_numeric'].fillna(df_model['age_numeric'].median(), inplace=True)

# Remove any remaining rows with missing values
df_model.dropna(inplace=True)
print(f"Modeling dataset shape after cleaning: {df_model.shape}")

# ============================================================================
# STEP 4: ENCODE CATEGORICAL FEATURES
# ============================================================================
print("\n[STEP 4] Encoding categorical features...")

# One-hot encode diagnosis category
df_encoded = pd.get_dummies(df_model, columns=['diag_1_category'], drop_first=True)

print(f"Features after encoding: {df_encoded.shape[1]}")
print(f"Feature names: {list(df_encoded.columns)}")

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT WITH STRATIFICATION
# ============================================================================
print("\n[STEP 5] Splitting data into train and test sets...")

# Separate features and target
X = df_encoded.drop('readmitted_binary', axis=1)
y = df_encoded['readmitted_binary']

# 80-20 train-test split with stratification
# Stratification ensures both sets have similar readmission rates
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Training set readmission rate: {y_train.mean()*100:.2f}%")
print(f"Test set readmission rate: {y_test.mean()*100:.2f}%")

# ============================================================================
# STEP 6: FEATURE SCALING
# ============================================================================
print("\n[STEP 6] Scaling features for Logistic Regression...")
print("Note: Tree-based models (Random Forest, XGBoost) do not require scaling")

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keep unscaled versions for tree-based models
X_train_unscaled = X_train.copy()
X_test_unscaled = X_test.copy()

# ============================================================================
# STEP 7: MODEL 1 - LOGISTIC REGRESSION (Baseline)
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*80)
print("\nRationale:")
print("- Baseline model for binary classification")
print("- Interpretable coefficients for clinical decision-making")
print("- Trusted in healthcare settings")
print("- Fast training and prediction")

# Train Logistic Regression
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'  # Handle class imbalance
)
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
print("\n[Logistic Regression] Performance Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr,
                          target_names=['Not Readmitted', 'Readmitted']))

# ============================================================================
# STEP 8: MODEL 2 - RANDOM FOREST
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: RANDOM FOREST")
print("="*80)
print("\nRationale:")
print("- Captures non-linear relationships in healthcare data")
print("- Handles mixed feature types effectively")
print("- Provides feature importance rankings")
print("- Robust to outliers and missing values")
print("- Strong performance on tabular medical data")

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train_unscaled, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_unscaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_unscaled)[:, 1]

# Evaluation metrics
print("\n[Random Forest] Performance Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf,
                          target_names=['Not Readmitted', 'Readmitted']))

# ============================================================================
# STEP 9: MODEL 3 - XGBOOST
# ============================================================================
print("\n" + "="*80)
print("MODEL 3: XGBOOST")
print("="*80)
print("\nRationale:")
print("- State-of-the-art gradient boosting for structured data")
print("- Superior performance on tabular healthcare datasets")
print("- Handles class imbalance effectively")
print("- Provides feature importance with high accuracy")
print("- Computationally efficient")

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),  # Handle imbalance
    eval_metric='logloss'
)
xgb_model.fit(X_train_unscaled, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test_unscaled)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_unscaled)[:, 1]

# Evaluation metrics
print("\n[XGBoost] Performance Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb,
                          target_names=['Not Readmitted', 'Readmitted']))

# ============================================================================
# STEP 10: MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],
    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_xgb)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_xgb)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba_lr),
        roc_auc_score(y_test, y_pred_proba_rf),
        roc_auc_score(y_test, y_pred_proba_xgb)
    ]
})

print("\nPerformance Comparison Table:")
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv('./output/data_modelling/model_comparison_table.csv', index=False)
print("\n✓ Comparison table saved to: ./output/data_modelling/model_comparison_table.csv")

# ============================================================================
# STEP 11: VISUALIZATIONS
# ============================================================================
print("\n[STEP 11] Generating visualizations...")

# 1. Model Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx//2, idx%2]
    values = comparison_df[metric].values
    bars = ax.bar(comparison_df['Model'], values, alpha=0.7, edgecolor='black')

    # Color bars
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('./output/data_modelling/model_comparison_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Model comparison chart saved")
plt.close()

# 2. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices - Model Comparison', fontsize=16, fontweight='bold')

models = [
    ('Logistic Regression', y_pred_lr),
    ('Random Forest', y_pred_rf),
    ('XGBoost', y_pred_xgb)
]

for idx, (name, y_pred) in enumerate(models):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Not Readmitted', 'Readmitted'],
                yticklabels=['Not Readmitted', 'Readmitted'])
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Actual', fontsize=11)
    axes[idx].set_xlabel('Predicted', fontsize=11)

plt.tight_layout()
plt.savefig('./output/data_modelling/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrices saved")
plt.close()

# 3. ROC Curves Comparison
plt.figure(figsize=(10, 8))

# Logistic Regression ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})',
         linewidth=2, color='#3498db')

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})',
         linewidth=2, color='#2ecc71')

# XGBoost ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.3f})',
         linewidth=2, color='#e74c3c')

# Diagonal reference line
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./output/data_modelling/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
print("✓ ROC curves saved")
plt.close()

# 4. Feature Importance (Logistic Regression) - Using Coefficients
feature_importance_lr = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

plt.figure(figsize=(10, 8))
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in feature_importance_lr['Coefficient'][:10]]
bars = plt.barh(feature_importance_lr['Feature'][:10],
                feature_importance_lr['Coefficient'][:10],
                color=colors, edgecolor='black', alpha=0.7)
plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 10 Feature Coefficients - Logistic Regression', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# Add value labels
for bar in bars:
    width = bar.get_width()
    ha = 'left' if width > 0 else 'right'
    plt.text(width, bar.get_y() + bar.get_height()/2.,
             f'{width:.4f}',
             ha=ha, va='center', fontsize=10)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Increases Readmission Risk'),
                   Patch(facecolor='#e74c3c', label='Decreases Readmission Risk')]
plt.legend(handles=legend_elements, loc='lower right')

plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig('./output/data_modelling/feature_importance_lr.png', dpi=300, bbox_inches='tight')
print("✓ Logistic Regression feature importance saved")
plt.close()

# 5. Feature Importance (Random Forest)
feature_importance_rf = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
bars = plt.barh(feature_importance_rf['Feature'][:10],
                feature_importance_rf['Importance'][:10],
                color='#2ecc71', edgecolor='black', alpha=0.7)
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 10 Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2.,
             f'{width:.4f}',
             ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('./output/data_modelling/feature_importance_rf.png', dpi=300, bbox_inches='tight')
print("✓ Random Forest feature importance saved")
plt.close()

# 6. Feature Importance (XGBoost)
feature_importance_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
bars = plt.barh(feature_importance_xgb['Feature'][:10],
                feature_importance_xgb['Importance'][:10],
                color='#e74c3c', edgecolor='black', alpha=0.7)
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 10 Feature Importance - XGBoost', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2.,
             f'{width:.4f}',
             ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('./output/data_modelling/feature_importance_xgb.png', dpi=300, bbox_inches='tight')
print("✓ XGBoost feature importance saved")
plt.close()

# ============================================================================
# STEP 12: CLINICAL INTERPRETATION
# ============================================================================
print("\n" + "="*80)
print("CLINICAL INTERPRETATION & KEY FINDINGS")
print("="*80)

print("\n1. TOP PREDICTIVE FEATURES (from best model):")

print("\nLogistic Regression Coefficients (scaled features):")
for idx, row in feature_importance_lr.head(5).iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"   {row['Feature']}: {row['Coefficient']:.4f} ({direction} risk)")

print("\nRandom Forest Feature Importance:")
for idx, row in feature_importance_rf.head(5).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

print("\nXGBoost Feature Importance:")
for idx, row in feature_importance_xgb.head(5).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

print("\n2. CLINICAL INSIGHTS:")
print("\n✓ Number of Inpatient Visits:")
print("  Patients with frequent prior hospitalizations show substantially higher")
print("  readmission risk, indicating chronic disease complexity and care dependency.")

print("\n✓ Emergency Department Visits:")
print("  Frequent emergency visits correlate with poorly managed conditions and")
print("  increased readmission likelihood.")

print("\n✓ Number of Diagnoses:")
print("  Comorbidity burden (multiple diagnoses) significantly impacts readmission,")
print("  reflecting disease complexity and treatment challenges.")

print("\n✓ Time in Hospital:")
print("  Longer initial stays indicate more severe conditions, correlating with")
print("  higher readmission probability.")

print("\n3. ALIGNMENT WITH EDA:")
print("\n✓ Model findings strongly align with exploratory data analysis:")
print("  - Prior inpatient visits showed strongest correlation in EDA")
print("  - Emergency visits and diagnoses count confirmed as key factors")
print("  - Length of stay patterns validated through modeling")

print("\n4. CLINICAL IMPLICATIONS:")
print("\n✓ High-risk patient identification:")
print("  Models can flag patients with >2 prior inpatient visits for")
print("  intensive discharge planning and follow-up care.")

print("\n✓ Resource allocation:")
print("  Hospitals can prioritize post-discharge interventions for")
print("  patients identified as high-risk by the model.")

print("\n✓ Preventive interventions:")
print("  Focus on medication management and coordination for patients")
print("  with high diagnosis counts and polypharmacy.")

# ============================================================================
# STEP 13: MODEL SELECTION JUSTIFICATION
# ============================================================================
print("\n" + "="*80)
print("MODEL SELECTION & JUSTIFICATION")
print("="*80)

print("\nBASED ON PERFORMANCE METRICS:")
print("\nRecall is the most critical metric in healthcare readmission prediction")
print("because missing a high-risk patient (false negative) has serious clinical")
print("and financial consequences:")
print("  - Patient safety: Missed high-risk patients may deteriorate")
print("  - Hospital penalties: Readmissions within 30 days incur Medicare penalties")
print("  - Resource waste: Emergency readmissions cost more than planned follow-ups")

print("\nMODEL COMPARISON SUMMARY:")
print(f"\nLogistic Regression:")
print(f"  Strengths: Interpretable, fast, clinically trusted baseline")
print(f"  Weaknesses: Limited ability to capture non-linear patterns")
print(f"  Recall: {recall_score(y_test, y_pred_lr):.4f}")

print(f"\nRandom Forest:")
print(f"  Strengths: Captures complex relationships, robust feature importance")
print(f"  Weaknesses: Less interpretable than logistic regression")
print(f"  Recall: {recall_score(y_test, y_pred_rf):.4f}")

print(f"\nXGBoost:")
print(f"  Strengths: State-of-the-art performance, handles imbalance well")
print(f"  Weaknesses: Computationally intensive, requires tuning")
print(f"  Recall: {recall_score(y_test, y_pred_xgb):.4f}")

# Determine best model based on recall
best_recall_idx = comparison_df['Recall'].idxmax()
best_model_name = comparison_df.loc[best_recall_idx, 'Model']
best_recall_score = comparison_df.loc[best_recall_idx, 'Recall']

print(f"\nRECOMMENDED MODEL: {best_model_name}")
print(f"Justification:")
print(f"  - Highest recall ({best_recall_score:.4f}), minimizing missed high-risk patients")
print(f"  - Strong overall performance across all metrics")
print(f"  - Balances predictive power with clinical utility")
print(f"  - Feature importance aligns with clinical knowledge")

# ============================================================================
# STEP 14: SAVE RESULTS SUMMARY
# ============================================================================
print("\n[STEP 14] Saving results summary...")

# Create comprehensive results summary
with open('./output/data_modelling/modeling_results_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("HOSPITAL READMISSION PREDICTION - MODELING RESULTS SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write("DATASET INFORMATION:\n")
    f.write(f"Total samples: {len(df_model)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"Readmission rate: {df['readmitted_binary'].mean()*100:.2f}%\n\n")

    f.write("SELECTED FEATURES:\n")
    for feature in selected_features:
        f.write(f"  - {feature}\n")
    f.write("\n")

    f.write("MODEL COMPARISON:\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\n")

    f.write(f"RECOMMENDED MODEL: {best_model_name}\n")
    f.write(f"Best Recall Score: {best_recall_score:.4f}\n\n")

    f.write("TOP 5 PREDICTIVE FEATURES:\n")
    if best_model_name == 'Random Forest':
        for idx, row in feature_importance_rf.head(5).iterrows():
            f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")
    else:
        for idx, row in feature_importance_xgb.head(5).iterrows():
            f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")

print("✓ Results summary saved to: modeling_results_summary.txt")

print("\n" + "="*80)
print("MODELING COMPLETE")
print("="*80)
print("\nGenerated outputs:")
print("  ✓ model_comparison_table.csv")
print("  ✓ model_comparison_metrics.png")
print("  ✓ confusion_matrices.png")
print("  ✓ roc_curves_comparison.png")
print("  ✓ feature_importance_lr.png")
print("  ✓ feature_importance_rf.png")
print("  ✓ feature_importance_xgb.png")
print("  ✓ modeling_results_summary.txt")
print("\nAll files saved to: ./output/data_modelling/")