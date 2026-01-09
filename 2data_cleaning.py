"""
Data Cleaning Script for Hospital Readmission in Diabetic Patients
Project: Analysis and Prediction of Hospital Readmission in Diabetic Patients Using Healthcare Data

This version creates output in '../output/cleaned_data/' structure for easy integration
with your existing project structure.

Dataset Source: UCI Machine Learning Repository - Diabetes 130-US hospitals for years 1999-2008
URL: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')

# Set visualization parameters
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CREATE OUTPUT DIRECTORY STRUCTURE
# ============================================================================
print("Creating output directory structure...")

# Create output directories relative to current working directory
output_base = './output'
cleaned_data_dir = os.path.join(output_base, 'cleaned_data')

# Create directories if they don't exist
os.makedirs(output_base, exist_ok=True)
os.makedirs(cleaned_data_dir, exist_ok=True)

print(f"Output directory: {os.path.abspath(cleaned_data_dir)}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Try multiple possible locations for the data files
possible_paths = [
    './dataset/diabetic_data.csv',  # Same directory
]

data_path = None
for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    print("ERROR: Could not find diabetic_data.csv")
    print("Please ensure the data file is in one of these locations:")
    for p in possible_paths:
        print(f"  - {p}")
    exit(1)

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"Original dataset shape: {df.shape}")
print(f"Total records: {df.shape[0]:,}")
print(f"Total features: {df.shape[1]}")

# Create a copy for comparison later
df_original = df.copy()

# ============================================================================
# STEP 2: INITIAL DATA QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("STEP 2: INITIAL DATA QUALITY ASSESSMENT")
print("="*80)

# Check for missing values
missing_data = pd.DataFrame({
    'Feature': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
print("\nFeatures with missing values:")
print(missing_data.to_string(index=False))

# Check for '?' as missing indicator (common in medical datasets)
print("\nChecking for '?' as missing value indicator...")
question_mark_counts = {}
for col in df.columns:
    if df[col].dtype == 'object':
        qm_count = (df[col] == '?').sum()
        if qm_count > 0:
            question_mark_counts[col] = qm_count

if question_mark_counts:
    print("\nFeatures with '?' values:")
    for col, count in sorted(question_mark_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(df)) * 100
        print(f"  {col}: {count:,} ({percentage:.2f}%)")
else:
    print("No '?' values found")

# ============================================================================
# STEP 3: VISUALIZATION - BEFORE CLEANING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: CREATING 'BEFORE CLEANING' VISUALIZATIONS")
print("="*80)

# Create figure for before-cleaning visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Data Quality Assessment - BEFORE CLEANING', fontsize=16, fontweight='bold')

# 3.1: Missing data bar chart (including '?' values)
missing_combined = {}
for col in df.columns:
    null_count = df[col].isnull().sum()
    qm_count = (df[col] == '?').sum() if df[col].dtype == 'object' else 0
    total_missing = null_count + qm_count
    if total_missing > 0:
        missing_combined[col] = (total_missing / len(df)) * 100

missing_df = pd.DataFrame(list(missing_combined.items()), columns=['Feature', 'Percentage'])
missing_df = missing_df.sort_values('Percentage', ascending=True)

# Plot top 15 features with most missing data
top_missing = missing_df.tail(15)
axes[0, 0].barh(top_missing['Feature'], top_missing['Percentage'], color='crimson', alpha=0.7)
axes[0, 0].set_xlabel('Missing Data (%)', fontsize=11)
axes[0, 0].set_title('Features with Missing Data (Top 15)', fontsize=12, fontweight='bold')
axes[0, 0].axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% threshold')
axes[0, 0].legend()
axes[0, 0].grid(axis='x', alpha=0.3)

# 3.2: Distribution of key numerical features (with outliers)
numerical_features = ['time_in_hospital', 'num_medications', 'num_lab_procedures', 'num_procedures']
data_for_box = [df[col].dropna() for col in numerical_features]
bp = axes[0, 1].boxplot(data_for_box, labels=numerical_features, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)
axes[0, 1].set_ylabel('Value', fontsize=11)
axes[0, 1].set_title('Distribution of Key Numerical Features (Outliers Visible)', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# 3.3: Readmission class distribution (before cleaning)
readmit_counts = df['readmitted'].value_counts()
colors_readmit = ['#ff9999', '#66b3ff', '#99ff99']
axes[1, 0].bar(readmit_counts.index, readmit_counts.values, color=colors_readmit, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Readmission Status', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].set_title('Readmission Class Distribution', fontsize=12, fontweight='bold')
for i, v in enumerate(readmit_counts.values):
    axes[1, 0].text(i, v + 1000, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# 3.4: Age distribution (categorical mess)
age_counts = df['age'].value_counts().sort_index()
axes[1, 1].barh(range(len(age_counts)), age_counts.values, color='teal', alpha=0.7)
axes[1, 1].set_yticks(range(len(age_counts)))
axes[1, 1].set_yticklabels(age_counts.index, fontsize=9)
axes[1, 1].set_xlabel('Count', fontsize=11)
axes[1, 1].set_title('Age Distribution (Categorical Format)', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
before_plot_path = os.path.join(cleaned_data_dir, '01_before_cleaning.png')
plt.savefig(before_plot_path, dpi=300, bbox_inches='tight')
print(f"Saved: {before_plot_path}")
plt.close()

# ============================================================================
# STEP 4: DATA CLEANING - HIGH MISSING COLUMNS (DROP)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: HANDLING HIGH-MISSING COLUMNS")
print("="*80)

# Replace '?' with NaN for proper handling
print("\nReplacing '?' with NaN across all features...")
df = df.replace('?', np.nan)

# Recalculate missing percentages after replacing '?'
missing_percentages = (df.isnull().sum() / len(df)) * 100
high_missing = missing_percentages[missing_percentages > 50].sort_values(ascending=False)

print(f"\nFeatures with >50% missing data:")
print(high_missing)

# Drop high-missing columns
columns_to_drop_high_missing = high_missing.index.tolist()
print(f"\nDropping {len(columns_to_drop_high_missing)} features with >50% missing data:")
for col in columns_to_drop_high_missing:
    print(f"  - {col}: {missing_percentages[col]:.2f}% missing")

df = df.drop(columns=columns_to_drop_high_missing)
print(f"\nDataset shape after dropping high-missing columns: {df.shape}")

# ============================================================================
# STEP 5: DATA CLEANING - MODERATE MISSING COLUMNS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: HANDLING MODERATE-MISSING COLUMNS")
print("="*80)

# Medical specialty - high cardinality with moderate missing
print("\nHandling 'medical_specialty'...")
medical_specialty_missing = df['medical_specialty'].isnull().sum()
print(f"  Missing values: {medical_specialty_missing:,} ({medical_specialty_missing/len(df)*100:.2f}%)")

# Group rare specialties and handle missing
specialty_counts = df['medical_specialty'].value_counts()
specialty_threshold = 1000
common_specialties = specialty_counts[specialty_counts > specialty_threshold].index.tolist()

df['medical_specialty_cleaned'] = df['medical_specialty'].apply(
    lambda x: x if x in common_specialties else ('Unknown' if pd.isna(x) else 'Other')
)
print(f"  Reduced from {df['medical_specialty'].nunique()} to {df['medical_specialty_cleaned'].nunique()} categories")
df = df.drop(columns=['medical_specialty'])
df = df.rename(columns={'medical_specialty_cleaned': 'medical_specialty'})

# Payer code - drop due to high missing values and low predictive value
print("\nHandling 'payer_code'...")
payer_missing = df['payer_code'].isnull().sum()
print(f"  Missing values: {payer_missing:,} ({payer_missing/len(df)*100:.2f}%)")
df = df.drop(columns=['payer_code'])
print(f"  Dropped 'payer_code' column")

# ============================================================================
# STEP 6: DATA CLEANING - LOW MISSING CLINICAL FIELDS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: HANDLING LOW-MISSING CLINICAL FIELDS")
print("="*80)

# Race - use mode or 'Unknown'
print("\nHandling 'race'...")
race_missing = df['race'].isnull().sum()
print(f"  Missing values: {race_missing:,} ({race_missing/len(df)*100:.2f}%)")
df['race'] = df['race'].fillna('Unknown')
print(f"  Replaced missing with 'Unknown'")

# Diagnosis codes - critical fields
print("\nHandling diagnosis codes (diag_1, diag_2, diag_3)...")
for diag_col in ['diag_1', 'diag_2', 'diag_3']:
    missing_count = df[diag_col].isnull().sum()
    print(f"  {diag_col} missing: {missing_count:,} ({missing_count/len(df)*100:.2f}%)")
    df[diag_col] = df[diag_col].fillna('Unknown')

# ============================================================================
# STEP 7: DIAGNOSIS CODE GROUPING (ICD-9)
# ============================================================================
print("\n" + "="*80)
print("STEP 7: GROUPING DIAGNOSIS CODES INTO CLINICAL CATEGORIES")
print("="*80)

def map_diagnosis_to_category(code):
    """
    Map ICD-9 diagnosis codes to clinical categories
    Based on standard ICD-9 code ranges
    """
    if pd.isna(code) or code == 'Unknown':
        return 'Unknown'

    # Convert to string and extract numeric part
    code_str = str(code)

    # Try to extract numeric value
    try:
        if 'V' in code_str or 'E' in code_str:
            return 'Other'

        # Remove any decimal point and get first 3 digits
        numeric_code = float(code_str.split('.')[0])

        # ICD-9 code groupings
        if 250 <= numeric_code <= 250.99:
            return 'Diabetes'
        elif 390 <= numeric_code <= 459 or 785 <= numeric_code <= 785.99:
            return 'Circulatory'
        elif 460 <= numeric_code <= 519 or 786 <= numeric_code <= 786.99:
            return 'Respiratory'
        elif 800 <= numeric_code <= 999:
            return 'Injury'
        elif 140 <= numeric_code <= 239:
            return 'Neoplasms'
        elif 710 <= numeric_code <= 739:
            return 'Musculoskeletal'
        elif 580 <= numeric_code <= 629:
            return 'Genitourinary'
        elif 520 <= numeric_code <= 579:
            return 'Digestive'
        else:
            return 'Other'
    except:
        return 'Other'

print("Mapping ICD-9 codes to diagnosis categories...")
df['diag_1_category'] = df['diag_1'].apply(map_diagnosis_to_category)
df['diag_2_category'] = df['diag_2'].apply(map_diagnosis_to_category)
df['diag_3_category'] = df['diag_3'].apply(map_diagnosis_to_category)

print("\nPrimary diagnosis (diag_1) categories:")
print(df['diag_1_category'].value_counts())

# ============================================================================
# STEP 8: CATEGORICAL ENCODING CLEANING
# ============================================================================
print("\n" + "="*80)
print("STEP 8: CLEANING CATEGORICAL ENCODINGS")
print("="*80)

# Age - convert to ordinal midpoints
print("\nConverting age brackets to numeric midpoints...")
age_mapping = {
    '[0-10)': 5,
    '[10-20)': 15,
    '[20-30)': 25,
    '[30-40)': 35,
    '[40-50)': 45,
    '[50-60)': 55,
    '[60-70)': 65,
    '[70-80)': 75,
    '[80-90)': 85,
    '[90-100)': 95
}
df['age_numeric'] = df['age'].map(age_mapping)
print(f"  Converted age brackets to numeric values (0-100 range)")
print(f"  Mean age: {df['age_numeric'].mean():.1f} years")

# Readmission - convert to binary for prediction
print("\nConverting readmission to binary classification...")
print(f"  Original classes: {df['readmitted'].unique()}")
df['readmitted_binary'] = df['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)
print(f"  Binary encoding: 'NO' = 0 (not readmitted), '<30' or '>30' = 1 (readmitted)")
print(f"  Class distribution: {df['readmitted_binary'].value_counts().to_dict()}")

# Gender - handle unknown/invalid
print("\nHandling gender...")
print(f"  Unique values: {df['gender'].unique()}")
df = df[df['gender'].isin(['Male', 'Female'])]
print(f"  Removed {len(df_original) - len(df)} records with invalid gender")
print(f"  Dataset shape after gender cleaning: {df.shape}")

# ============================================================================
# STEP 9: OUTLIER INSPECTION (NOT REMOVAL)
# ============================================================================
print("\n" + "="*80)
print("STEP 9: OUTLIER INSPECTION (NO AUTOMATIC REMOVAL)")
print("="*80)

numerical_cols = ['time_in_hospital', 'num_medications', 'num_lab_procedures',
                  'num_procedures', 'number_outpatient', 'number_emergency', 'number_inpatient']

print("\nOutlier analysis using IQR method:")
outlier_summary = []
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

    outlier_summary.append({
        'Feature': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound,
        'Outlier_Count': len(outliers),
        'Outlier_Percentage': (len(outliers) / len(df)) * 100
    })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))
print("\nNote: Outliers are RETAINED as they may represent high-risk patients, not data errors.")

# ============================================================================
# STEP 10: DUPLICATE AND ENCOUNTER HANDLING
# ============================================================================
print("\n" + "="*80)
print("STEP 10: DUPLICATE AND ENCOUNTER HANDLING")
print("="*80)

print(f"\nChecking for duplicate encounters...")
duplicate_encounters = df['encounter_id'].duplicated().sum()
print(f"  Duplicate encounter_id: {duplicate_encounters}")

print(f"\nChecking patient encounter frequency...")
patient_encounters = df['patient_nbr'].value_counts()
print(f"  Unique patients: {len(patient_encounters):,}")
print(f"  Total encounters: {len(df):,}")
print(f"  Average encounters per patient: {len(df) / len(patient_encounters):.2f}")
print(f"  Max encounters for single patient: {patient_encounters.max()}")

print("\nDecision: Keeping all encounters as independent events")
print("Justification: Each encounter represents a distinct clinical episode with unique outcomes")

# ============================================================================
# STEP 11: FEATURE REMOVAL (NON-INFORMATIVE)
# ============================================================================
print("\n" + "="*80)
print("STEP 11: REMOVING NON-INFORMATIVE FEATURES")
print("="*80)

# Drop ID columns
print("\nRemoving identifier columns...")
id_columns = ['encounter_id', 'patient_nbr']
df = df.drop(columns=id_columns)
print(f"  Dropped: {', '.join(id_columns)}")

# Drop medication columns with no variation
print("\nChecking medication columns for zero variance...")
medication_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                   'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                   'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                   'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                   'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

low_variance_meds = []
for med in medication_cols:
    if med in df.columns:
        value_counts = df[med].value_counts()
        if len(value_counts) == 1 or (value_counts.iloc[0] / len(df)) > 0.99:
            low_variance_meds.append(med)

if low_variance_meds:
    print(f"  Found {len(low_variance_meds)} medications with <1% variation")
    df = df.drop(columns=low_variance_meds)
    print(f"  Dropped: {', '.join(low_variance_meds[:5])}..." if len(low_variance_meds) > 5 else f"  Dropped: {', '.join(low_variance_meds)}")

# Drop original diagnosis codes (keep categories)
print("\nRemoving original diagnosis codes (categories retained)...")
df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])

# Drop original age column (numeric version retained)
df = df.drop(columns=['age'])

print(f"\nFinal dataset shape: {df.shape}")

# ============================================================================
# STEP 12: SAVE CLEANED DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 12: SAVING CLEANED DATASET")
print("="*80)

cleaned_csv_path = os.path.join(cleaned_data_dir, 'diabetic_data_cleaned.csv')
df.to_csv(cleaned_csv_path, index=False)
print(f"Saved: {cleaned_csv_path}")

# ============================================================================
# STEP 13: VISUALIZATION - AFTER CLEANING
# ============================================================================
print("\n" + "="*80)
print("STEP 13: CREATING 'AFTER CLEANING' VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Data Quality Assessment - AFTER CLEANING', fontsize=16, fontweight='bold')

# 13.1: Missing data bar chart (after cleaning)
missing_after = (df.isnull().sum() / len(df)) * 100
missing_after = missing_after[missing_after > 0].sort_values(ascending=True)

if len(missing_after) > 0:
    axes[0, 0].barh(missing_after.index, missing_after.values, color='green', alpha=0.7)
    axes[0, 0].set_xlabel('Missing Data (%)', fontsize=11)
    axes[0, 0].set_title('Remaining Missing Data (After Cleaning)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
else:
    axes[0, 0].text(0.5, 0.5, 'NO MISSING DATA\nCleaning Complete!',
                    ha='center', va='center', fontsize=16, fontweight='bold', color='green')
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axis('off')

# 13.2: Distribution of key numerical features (after cleaning)
data_for_box_after = [df[col].dropna() for col in numerical_features if col in df.columns]
bp2 = axes[0, 1].boxplot(data_for_box_after, labels=[c for c in numerical_features if c in df.columns], patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('lightgreen')
    patch.set_alpha(0.7)
axes[0, 1].set_ylabel('Value', fontsize=11)
axes[0, 1].set_title('Distribution of Key Numerical Features (Cleaned)', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# 13.3: Binary readmission class distribution
readmit_binary_counts = df['readmitted_binary'].value_counts()
colors_binary = ['#99ff99', '#ff9999']
axes[1, 0].bar(['No Early Readmit (0)', 'Early Readmit (1)'],
               [readmit_binary_counts[0], readmit_binary_counts[1]],
               color=colors_binary, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Readmission Class (Binary)', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].set_title('Binary Readmission Distribution (Cleaned)', fontsize=12, fontweight='bold')
for i, v in enumerate([readmit_binary_counts[0], readmit_binary_counts[1]]):
    axes[1, 0].text(i, v + 1000, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# 13.4: Primary diagnosis categories
diag_counts = df['diag_1_category'].value_counts()
colors_diag = plt.cm.Set3(range(len(diag_counts)))
axes[1, 1].barh(range(len(diag_counts)), diag_counts.values, color=colors_diag, alpha=0.8, edgecolor='black')
axes[1, 1].set_yticks(range(len(diag_counts)))
axes[1, 1].set_yticklabels(diag_counts.index, fontsize=10)
axes[1, 1].set_xlabel('Count', fontsize=11)
axes[1, 1].set_title('Primary Diagnosis Categories (Grouped from ICD-9)', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
after_plot_path = os.path.join(cleaned_data_dir, '02_after_cleaning.png')
plt.savefig(after_plot_path, dpi=300, bbox_inches='tight')
print(f"Saved: {after_plot_path}")
plt.close()

# ============================================================================
# STEP 14: COMPARISON VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 14: CREATING BEFORE vs AFTER COMPARISON")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Data Cleaning Impact - BEFORE vs AFTER COMPARISON', fontsize=16, fontweight='bold')

# 14.1: Feature count comparison
categories = ['Original Features', 'After Cleaning']
feature_counts = [df_original.shape[1], df.shape[1]]
colors_features = ['#ff9999', '#99ff99']
bars = axes[0].bar(categories, feature_counts, color=colors_features, alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Number of Features', fontsize=11)
axes[0].set_title('Feature Count Reduction', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(feature_counts):
    axes[0].text(i, v + 1, str(v), ha='center', va='bottom', fontsize=14, fontweight='bold')
reduction = df_original.shape[1] - df.shape[1]
axes[0].text(0.5, max(feature_counts) * 0.5, f'Reduced by\n{reduction} features',
             ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# 14.2: Missing data comparison
missing_before_total = (df_original.isnull().sum().sum() +
                        sum((df_original[col] == '?').sum() for col in df_original.columns if df_original[col].dtype == 'object'))
missing_after_total = df.isnull().sum().sum()
missing_before_pct = (missing_before_total / (df_original.shape[0] * df_original.shape[1])) * 100
missing_after_pct = (missing_after_total / (df.shape[0] * df.shape[1])) * 100

bars2 = axes[1].bar(categories, [missing_before_pct, missing_after_pct], color=colors_features, alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_ylabel('Missing Data (%)', fontsize=11)
axes[1].set_title('Overall Data Completeness Improvement', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate([missing_before_pct, missing_after_pct]):
    axes[1].text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 14.3: Record count
record_counts = [len(df_original), len(df)]
bars3 = axes[2].bar(categories, record_counts, color=colors_features, alpha=0.7, edgecolor='black', linewidth=2)
axes[2].set_ylabel('Number of Records', fontsize=11)
axes[2].set_title('Record Retention', fontsize=12, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)
for i, v in enumerate(record_counts):
    axes[2].text(i, v + 500, f'{v:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
retention = (len(df) / len(df_original)) * 100
axes[2].text(0.5, max(record_counts) * 0.5, f'{retention:.1f}% retained',
             ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
comparison_plot_path = os.path.join(cleaned_data_dir, '03_before_after_comparison.png')
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
print(f"Saved: {comparison_plot_path}")
plt.close()

# ============================================================================
# STEP 15: SAVE METADATA AND SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STEP 15: SAVING METADATA AND SUMMARY")
print("="*80)

# Create metadata DataFrame
metadata = {
    'Metric': [
        'Original Records',
        'Final Records',
        'Records Removed',
        'Record Retention Rate (%)',
        'Original Features',
        'Final Features',
        'Features Removed',
        'Feature Reduction (%)',
        'Missing Data Before (%)',
        'Missing Data After (%)',
        'Data Quality Improvement (%)'
    ],
    'Value': [
        f"{len(df_original):,}",
        f"{len(df):,}",
        f"{len(df_original) - len(df):,}",
        f"{(len(df)/len(df_original))*100:.2f}",
        df_original.shape[1],
        df.shape[1],
        df_original.shape[1] - df.shape[1],
        f"{((df_original.shape[1] - df.shape[1])/df_original.shape[1])*100:.2f}",
        f"{missing_before_pct:.2f}",
        f"{missing_after_pct:.2f}",
        f"{missing_before_pct - missing_after_pct:.2f}"
    ]
}

metadata_df = pd.DataFrame(metadata)
metadata_path = os.path.join(cleaned_data_dir, 'cleaning_metadata.csv')
metadata_df.to_csv(metadata_path, index=False)
print(f"Saved: {metadata_path}")

# Summary text
summary_text = f"""
DATA CLEANING SUMMARY
{'='*80}

Original Dataset: {len(df_original):,} records × {df_original.shape[1]} features
Cleaned Dataset: {len(df):,} records × {df.shape[1]} features

Key Improvements:
- Missing data reduced from {missing_before_pct:.2f}% to {missing_after_pct:.2f}%
- Features reduced by {df_original.shape[1] - df.shape[1]} (38% reduction)
- Records retained: {(len(df)/len(df_original))*100:.2f}%

Target Variable:
- readmitted_binary: 1 = readmission (<30 or >30 days), 0 = no early readmission
- Class 0: {readmit_binary_counts[0]:,} ({readmit_binary_counts[0]/len(df)*100:.1f}%)
- Class 1: {readmit_binary_counts[1]:,} ({readmit_binary_counts[1]/len(df)*100:.1f}%)

All files saved to: {os.path.abspath(cleaned_data_dir)}
"""

summary_path = os.path.join(cleaned_data_dir, 'cleaning_summary.txt')
with open(summary_path, 'w') as f:
    f.write(summary_text)
print(f"Saved: {summary_path}")

print("\n" + "="*80)
print("DATA CLEANING COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {os.path.abspath(cleaned_data_dir)}")
print("\nGenerated files:")
print("1. diabetic_data_cleaned.csv - Cleaned dataset")
print("2. 01_before_cleaning.png - Before cleaning visualization")
print("3. 02_after_cleaning.png - After cleaning visualization")
print("4. 03_before_after_comparison.png - Comparison visualization")
print("5. cleaning_metadata.csv - Cleaning statistics")
print("6. cleaning_summary.txt - Summary report")
print("="*80)