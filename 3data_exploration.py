"""
Analysis and Prediction of Hospital Readmission in Diabetic Patients Using Healthcare Data
Exploratory Data Analysis Script - CLEANED DATASET VERSION

This script performs comprehensive exploratory analysis on the cleaned diabetic patient
readmission dataset, including target variable exploration, univariate analysis, bivariate
analysis, correlation studies, and patient utilization pattern analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create output directory if it doesn't exist
output_dir = './output/data_exploration/'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Load the data
print("Loading cleaned dataset...")
diabetic_data = pd.read_csv('./dataset/diabetic_data_cleaned.csv')

print(f"Dataset shape: {diabetic_data.shape}")
print(f"\nColumns in cleaned dataset:")
print(diabetic_data.columns.tolist())
print(f"\nFirst few rows:")
print(diabetic_data.head())

# Create age groups from age_numeric for better visualization
def create_age_group(age_numeric):
    """Convert numeric age to age group string"""
    if age_numeric < 10:
        return '[0-10)'
    elif age_numeric < 20:
        return '[10-20)'
    elif age_numeric < 30:
        return '[20-30)'
    elif age_numeric < 40:
        return '[30-40)'
    elif age_numeric < 50:
        return '[40-50)'
    elif age_numeric < 60:
        return '[50-60)'
    elif age_numeric < 70:
        return '[60-70)'
    elif age_numeric < 80:
        return '[70-80)'
    elif age_numeric < 90:
        return '[80-90)'
    else:
        return '[90-100)'

diabetic_data['age_group'] = diabetic_data['age_numeric'].apply(create_age_group)

# ==============================================================================
# SECTION 1: TARGET VARIABLE EXPLORATION (READMISSION)
# ==============================================================================
print("\n" + "="*80)
print("SECTION 1: TARGET VARIABLE EXPLORATION (READMISSION)")
print("="*80)

# Distribution of readmitted variable
print("\nReadmission Distribution:")
print(diabetic_data['readmitted'].value_counts())
print("\nReadmission Percentages:")
print(diabetic_data['readmitted'].value_counts(normalize=True) * 100)

# Check binary readmission variable (should already exist)
print(f"\nReadmitted within 30 days (binary): {diabetic_data['readmitted_binary'].sum()} ({diabetic_data['readmitted_binary'].mean()*100:.2f}%)")

# Visualization 1: Distribution of readmission
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original distribution (3 classes)
readmit_counts = diabetic_data['readmitted'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#f39c12']
axes[0].bar(readmit_counts.index, readmit_counts.values, color=colors, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Readmission Status', fontweight='bold')
axes[0].set_ylabel('Number of Patients', fontweight='bold')
axes[0].set_title('Distribution of Readmission Status (3 Classes)', fontweight='bold', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(readmit_counts.values):
    axes[0].text(i, v + 1000, str(v), ha='center', fontweight='bold')

# Binary distribution
binary_counts = diabetic_data['readmitted_binary'].value_counts()
binary_labels = ['Not Readmitted\nWithin 30 Days', 'Readmitted\nWithin 30 Days']
colors_binary = ['#3498db', '#e74c3c']
axes[1].bar(binary_labels, [binary_counts[0], binary_counts[1]], color=colors_binary, alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Number of Patients', fontweight='bold')
axes[1].set_title('Distribution of Readmission Status (Binary)', fontweight='bold', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate([binary_counts[0], binary_counts[1]]):
    axes[1].text(i, v + 1000, f'{v}\n({v/len(diabetic_data)*100:.1f}%)', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}01_readmission_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 01_readmission_distribution.png")

# Visualization 2: Readmission rate by age group
print("\nAnalyzing readmission by age group...")
age_readmit = diabetic_data.groupby('age_group')['readmitted_binary'].agg(['sum', 'count', 'mean'])
age_readmit.columns = ['Readmitted', 'Total', 'Rate']
age_readmit['Rate'] = age_readmit['Rate'] * 100
age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
age_readmit = age_readmit.reindex(age_order)

print("\nReadmission Rate by Age Group:")
print(age_readmit)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Readmission counts
x_pos = np.arange(len(age_readmit))

axes[0].bar(x_pos, age_readmit['Total'], alpha=0.5, label='Total Patients', color='#3498db', edgecolor='black')
axes[0].bar(x_pos, age_readmit['Readmitted'], alpha=0.8, label='Readmitted <30 Days', color='#e74c3c', edgecolor='black')
axes[0].set_xlabel('Age Group', fontweight='bold')
axes[0].set_ylabel('Number of Patients', fontweight='bold')
axes[0].set_title('Readmission Counts by Age Group', fontweight='bold', fontsize=12)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(age_readmit.index, rotation=45)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Readmission rate percentage
axes[1].plot(x_pos, age_readmit['Rate'], marker='o', linewidth=2, markersize=8, color='#e74c3c')
axes[1].fill_between(x_pos, age_readmit['Rate'], alpha=0.3, color='#e74c3c')
axes[1].set_xlabel('Age Group', fontweight='bold')
axes[1].set_ylabel('Readmission Rate (%)', fontweight='bold')
axes[1].set_title('Readmission Rate by Age Group', fontweight='bold', fontsize=12)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(age_readmit.index, rotation=45)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=age_readmit['Rate'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {age_readmit["Rate"].mean():.2f}%')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{output_dir}02_readmission_by_age.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_readmission_by_age.png")

# Visualization 3: Readmission rate by number of inpatient visits
print("\nAnalyzing readmission by prior inpatient visits...")
# Cap at 5+ for better visualization
diabetic_data['inpatient_binned'] = diabetic_data['number_inpatient'].apply(lambda x: '5+' if x >= 5 else str(x))
inpatient_order = ['0', '1', '2', '3', '4', '5+']

inpatient_readmit = diabetic_data.groupby('inpatient_binned')['readmitted_binary'].agg(['sum', 'count', 'mean'])
inpatient_readmit.columns = ['Readmitted', 'Total', 'Rate']
inpatient_readmit['Rate'] = inpatient_readmit['Rate'] * 100
inpatient_readmit = inpatient_readmit.reindex(inpatient_order)

print("\nReadmission Rate by Prior Inpatient Visits:")
print(inpatient_readmit)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Stacked bar chart
axes[0].bar(inpatient_readmit.index, inpatient_readmit['Total'], alpha=0.5, label='Total Patients', color='#3498db', edgecolor='black')
axes[0].bar(inpatient_readmit.index, inpatient_readmit['Readmitted'], alpha=0.8, label='Readmitted <30 Days', color='#e74c3c', edgecolor='black')
axes[0].set_xlabel('Number of Prior Inpatient Visits', fontweight='bold')
axes[0].set_ylabel('Number of Patients', fontweight='bold')
axes[0].set_title('Readmission Counts by Prior Inpatient Visits', fontweight='bold', fontsize=12)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Readmission rate trend
axes[1].plot(range(len(inpatient_readmit)), inpatient_readmit['Rate'], marker='o', linewidth=3, markersize=10, color='#e74c3c')
axes[1].fill_between(range(len(inpatient_readmit)), inpatient_readmit['Rate'], alpha=0.3, color='#e74c3c')
axes[1].set_xlabel('Number of Prior Inpatient Visits', fontweight='bold')
axes[1].set_ylabel('Readmission Rate (%)', fontweight='bold')
axes[1].set_title('Readmission Rate by Prior Inpatient Visits (Clear Upward Trend)', fontweight='bold', fontsize=12)
axes[1].set_xticks(range(len(inpatient_readmit)))
axes[1].set_xticklabels(inpatient_readmit.index)
axes[1].grid(True, alpha=0.3)
for i, rate in enumerate(inpatient_readmit['Rate']):
    axes[1].annotate(f'{rate:.1f}%', xy=(i, rate), xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}03_readmission_by_inpatient_visits.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_readmission_by_inpatient_visits.png")

# ==============================================================================
# SECTION 2: UNIVARIATE ANALYSIS (NUMERICAL FEATURES)
# ==============================================================================
print("\n" + "="*80)
print("SECTION 2: UNIVARIATE ANALYSIS (NUMERICAL FEATURES)")
print("="*80)

# Key numerical features to analyze
numerical_features = ['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_diagnoses']

# Summary statistics
print("\nSummary Statistics for Key Numerical Features:")
print(diabetic_data[numerical_features].describe())

# Visualization 4: Histograms of numerical features
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

colors_hist = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for idx, feature in enumerate(numerical_features):
    # Histogram
    axes[idx].hist(diabetic_data[feature], bins=50, alpha=0.7, color=colors_hist[idx], edgecolor='black')
    axes[idx].set_xlabel(feature.replace('_', ' ').title(), fontweight='bold')
    axes[idx].set_ylabel('Frequency', fontweight='bold')
    axes[idx].set_title(f'Distribution of {feature.replace("_", " ").title()}', fontweight='bold', fontsize=11)
    axes[idx].grid(axis='y', alpha=0.3)

    # Add statistics
    mean_val = diabetic_data[feature].mean()
    median_val = diabetic_data[feature].median()
    axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    axes[idx].axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    axes[idx].legend()

    # Check for right skew
    skewness = diabetic_data[feature].skew()
    axes[idx].text(0.98, 0.97, f'Skewness: {skewness:.2f}',
                   transform=axes[idx].transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}04_univariate_histograms.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 04_univariate_histograms.png")

# Visualization 5: Boxplots split by readmission status
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, feature in enumerate(numerical_features):
    # Boxplot by readmission status
    data_to_plot = [
        diabetic_data[diabetic_data['readmitted_binary'] == 0][feature],
        diabetic_data[diabetic_data['readmitted_binary'] == 1][feature]
    ]

    bp = axes[idx].boxplot(data_to_plot, labels=['Not Readmitted', 'Readmitted <30d'],
                           patch_artist=True, widths=0.6,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))

    # Color the boxes differently
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')

    axes[idx].set_ylabel(feature.replace('_', ' ').title(), fontweight='bold')
    axes[idx].set_title(f'{feature.replace("_", " ").title()} by Readmission Status', fontweight='bold', fontsize=11)
    axes[idx].grid(axis='y', alpha=0.3)

    # Statistical test
    not_readmit = diabetic_data[diabetic_data['readmitted_binary'] == 0][feature]
    readmit = diabetic_data[diabetic_data['readmitted_binary'] == 1][feature]
    t_stat, p_value = stats.ttest_ind(not_readmit, readmit)

    axes[idx].text(0.98, 0.97, f'p-value: {p_value:.4f}',
                   transform=axes[idx].transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow' if p_value < 0.05 else 'white', alpha=0.7),
                   fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}05_univariate_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_univariate_boxplots.png")

# ==============================================================================
# SECTION 3: BIVARIATE ANALYSIS (CATEGORICAL FEATURES)
# ==============================================================================
print("\n" + "="*80)
print("SECTION 3: BIVARIATE ANALYSIS (CATEGORICAL FEATURES)")
print("="*80)

# Use the existing diag_1_category column
print("\nDiagnosis Category Distribution:")
print(diabetic_data['diag_1_category'].value_counts())

# Visualization 6: Readmission rate by primary diagnosis
diag_readmit = diabetic_data.groupby('diag_1_category')['readmitted_binary'].agg(['sum', 'count', 'mean'])
diag_readmit.columns = ['Readmitted', 'Total', 'Rate']
diag_readmit['Rate'] = diag_readmit['Rate'] * 100
diag_readmit = diag_readmit.sort_values('Rate', ascending=False)

print("\nReadmission Rate by Primary Diagnosis Category:")
print(diag_readmit)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Stacked bar chart
diag_categories = diag_readmit.index
x_pos = np.arange(len(diag_categories))

axes[0].bar(x_pos, diag_readmit['Total'], alpha=0.5, label='Total Patients', color='#3498db', edgecolor='black')
axes[0].bar(x_pos, diag_readmit['Readmitted'], alpha=0.8, label='Readmitted <30 Days', color='#e74c3c', edgecolor='black')
axes[0].set_xlabel('Primary Diagnosis Category', fontweight='bold')
axes[0].set_ylabel('Number of Patients', fontweight='bold')
axes[0].set_title('Readmission Counts by Primary Diagnosis', fontweight='bold', fontsize=12)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(diag_categories, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Readmission rate percentage
colors_bar = ['#e74c3c' if rate > diag_readmit['Rate'].mean() else '#3498db' for rate in diag_readmit['Rate']]
axes[1].barh(x_pos, diag_readmit['Rate'], color=colors_bar, alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Primary Diagnosis Category', fontweight='bold')
axes[1].set_xlabel('Readmission Rate (%)', fontweight='bold')
axes[1].set_title('Readmission Rate by Primary Diagnosis (Sorted)', fontweight='bold', fontsize=12)
axes[1].set_yticks(x_pos)
axes[1].set_yticklabels(diag_categories)
axes[1].axvline(x=diag_readmit['Rate'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {diag_readmit["Rate"].mean():.2f}%')
axes[1].legend()
axes[1].grid(axis='x', alpha=0.3)

for i, rate in enumerate(diag_readmit['Rate']):
    axes[1].text(rate + 0.1, i, f'{rate:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}06_readmission_by_diagnosis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 06_readmission_by_diagnosis.png")

# Visualization 7: Readmission rate by medical specialty (top 15)
print("\nAnalyzing readmission by medical specialty...")
specialty_readmit = diabetic_data[diabetic_data['medical_specialty'] != '?'].groupby('medical_specialty')['readmitted_binary'].agg(['sum', 'count', 'mean'])
specialty_readmit.columns = ['Readmitted', 'Total', 'Rate']
specialty_readmit['Rate'] = specialty_readmit['Rate'] * 100
specialty_readmit = specialty_readmit[specialty_readmit['Total'] >= 100]  # Filter for meaningful sample sizes
specialty_readmit = specialty_readmit.sort_values('Rate', ascending=False).head(15)

print("\nTop 15 Specialties by Readmission Rate (min 100 patients):")
print(specialty_readmit)

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(specialty_readmit))
colors_specialty = ['#e74c3c' if rate > specialty_readmit['Rate'].mean() else '#2ecc71' for rate in specialty_readmit['Rate']]

ax.barh(y_pos, specialty_readmit['Rate'], color=colors_specialty, alpha=0.7, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(specialty_readmit.index)
ax.set_xlabel('Readmission Rate (%)', fontweight='bold')
ax.set_ylabel('Medical Specialty', fontweight='bold')
ax.set_title('Top 15 Medical Specialties by Readmission Rate (Sample Size ≥ 100)', fontweight='bold', fontsize=12)
ax.axvline(x=specialty_readmit['Rate'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {specialty_readmit["Rate"].mean():.2f}%')
ax.legend()
ax.grid(axis='x', alpha=0.3)

for i, (rate, total) in enumerate(zip(specialty_readmit['Rate'], specialty_readmit['Total'])):
    ax.text(rate + 0.2, i, f'{rate:.1f}% (n={total})', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}07_readmission_by_specialty.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_readmission_by_specialty.png")

# Visualization 8: Readmission by discharge disposition
print("\nAnalyzing readmission by discharge disposition...")
discharge_readmit = diabetic_data.groupby('discharge_disposition_id')['readmitted_binary'].agg(['sum', 'count', 'mean'])
discharge_readmit.columns = ['Readmitted', 'Total', 'Rate']
discharge_readmit['Rate'] = discharge_readmit['Rate'] * 100
discharge_readmit = discharge_readmit[discharge_readmit['Total'] >= 50].sort_values('Rate', ascending=False).head(10)

print("\nTop 10 Discharge Dispositions by Readmission Rate (min 50 patients):")
print(discharge_readmit)

fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(discharge_readmit))
colors_discharge = plt.cm.RdYlGn_r(discharge_readmit['Rate'] / discharge_readmit['Rate'].max())

ax.bar(x_pos, discharge_readmit['Rate'], color=colors_discharge, alpha=0.7, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Disp {idx}' for idx in discharge_readmit.index], rotation=0)
ax.set_xlabel('Discharge Disposition ID', fontweight='bold')
ax.set_ylabel('Readmission Rate (%)', fontweight='bold')
ax.set_title('Top 10 Discharge Dispositions by Readmission Rate', fontweight='bold', fontsize=12)
ax.axhline(y=discharge_readmit['Rate'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {discharge_readmit["Rate"].mean():.2f}%')
ax.legend()
ax.grid(axis='y', alpha=0.3)

for i, (rate, total) in enumerate(zip(discharge_readmit['Rate'], discharge_readmit['Total'])):
    ax.text(i, rate + 0.5, f'{rate:.1f}%\n(n={total})', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}08_readmission_by_discharge.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 08_readmission_by_discharge.png")

# ==============================================================================
# SECTION 4: CORRELATION ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("SECTION 4: CORRELATION ANALYSIS & MULTICOLLINEARITY")
print("="*80)

# Select numerical features for correlation
corr_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                 'num_medications', 'number_outpatient', 'number_emergency',
                 'number_inpatient', 'number_diagnoses', 'age_numeric', 'readmitted_binary']

correlation_data = diabetic_data[corr_features].copy()

print("\nCorrelation Matrix:")
correlation_matrix = correlation_data.corr()
print(correlation_matrix)

# Visualization 9: Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)

ax.set_title('Correlation Heatmap of Numerical Features', fontweight='bold', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}09_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 09_correlation_heatmap.png")

# Identify strongest correlations with readmission
readmit_corr = correlation_matrix['readmitted_binary'].drop('readmitted_binary').sort_values(ascending=False)
print("\nCorrelations with Readmission (sorted by strength):")
print(readmit_corr)

# Visualization 10: Feature correlations with readmission
fig, ax = plt.subplots(figsize=(10, 6))
colors_corr = ['#e74c3c' if x > 0 else '#3498db' for x in readmit_corr.values]
y_pos = np.arange(len(readmit_corr))

ax.barh(y_pos, readmit_corr.values, color=colors_corr, alpha=0.7, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(readmit_corr.index)
ax.set_xlabel('Correlation Coefficient', fontweight='bold')
ax.set_ylabel('Feature', fontweight='bold')
ax.set_title('Feature Correlations with Readmission (<30 days)', fontweight='bold', fontsize=12)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(axis='x', alpha=0.3)

for i, v in enumerate(readmit_corr.values):
    ax.text(v + 0.002 if v > 0 else v - 0.002, i, f'{v:.4f}',
            va='center', ha='left' if v > 0 else 'right', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}10_readmission_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 10_readmission_correlations.png")

# Identify multicollinearity
print("\nMulticollinearity Check (pairs with |correlation| > 0.7):")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'Feature 1': correlation_matrix.columns[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    print(pd.DataFrame(high_corr_pairs))
else:
    print("No strong multicollinearity detected (threshold: 0.7)")

# ==============================================================================
# SECTION 5: PATIENT UTILIZATION PATTERNS (HIGH VALUE ANALYSIS)
# ==============================================================================
print("\n" + "="*80)
print("SECTION 5: PATIENT UTILIZATION PATTERNS")
print("="*80)

# Create utilization categories
diabetic_data['total_visits'] = (diabetic_data['number_outpatient'] +
                                  diabetic_data['number_emergency'] +
                                  diabetic_data['number_inpatient'])

# Categorize patients by utilization
diabetic_data['utilization_category'] = pd.cut(diabetic_data['total_visits'],
                                                bins=[-1, 0, 2, 5, 100],
                                                labels=['No Prior Visits', 'Low (1-2)', 'Medium (3-5)', 'High (6+)'])

# Visualization 11: Readmission by total prior visits
visits_readmit = diabetic_data.groupby('utilization_category')['readmitted_binary'].agg(['sum', 'count', 'mean'])
visits_readmit.columns = ['Readmitted', 'Total', 'Rate']
visits_readmit['Rate'] = visits_readmit['Rate'] * 100

print("\nReadmission Rate by Utilization Category:")
print(visits_readmit)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Counts
x_pos = np.arange(len(visits_readmit))
axes[0].bar(x_pos, visits_readmit['Total'], alpha=0.5, label='Total Patients', color='#3498db', edgecolor='black')
axes[0].bar(x_pos, visits_readmit['Readmitted'], alpha=0.8, label='Readmitted <30 Days', color='#e74c3c', edgecolor='black')
axes[0].set_xlabel('Prior Utilization Category', fontweight='bold')
axes[0].set_ylabel('Number of Patients', fontweight='bold')
axes[0].set_title('Patient Counts by Utilization Category', fontweight='bold', fontsize=12)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(visits_readmit.index, rotation=15)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Rate trend
axes[1].plot(x_pos, visits_readmit['Rate'], marker='o', linewidth=3, markersize=12, color='#e74c3c', label='Readmission Rate')
axes[1].fill_between(x_pos, visits_readmit['Rate'], alpha=0.3, color='#e74c3c')
axes[1].set_xlabel('Prior Utilization Category', fontweight='bold')
axes[1].set_ylabel('Readmission Rate (%)', fontweight='bold')
axes[1].set_title('Readmission Rate Increases with Prior Utilization', fontweight='bold', fontsize=12)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(visits_readmit.index, rotation=15)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

for i, rate in enumerate(visits_readmit['Rate']):
    axes[1].annotate(f'{rate:.1f}%', xy=(i, rate), xytext=(0, 10),
                     textcoords='offset points', ha='center', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{output_dir}11_utilization_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 11_utilization_patterns.png")

# Visualization 12: Detailed breakdown by visit type
visit_types = ['number_outpatient', 'number_emergency', 'number_inpatient']
visit_labels = ['Outpatient Visits', 'Emergency Visits', 'Inpatient Visits']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (visit_type, label) in enumerate(zip(visit_types, visit_labels)):
    # Cap for visualization
    diabetic_data[f'{visit_type}_capped'] = diabetic_data[visit_type].apply(lambda x: min(x, 5))

    visit_readmit = diabetic_data.groupby(f'{visit_type}_capped')['readmitted_binary'].agg(['sum', 'count', 'mean'])
    visit_readmit.columns = ['Readmitted', 'Total', 'Rate']
    visit_readmit['Rate'] = visit_readmit['Rate'] * 100

    x_pos = visit_readmit.index
    axes[idx].plot(x_pos, visit_readmit['Rate'], marker='o', linewidth=2, markersize=8, color='#e74c3c')
    axes[idx].fill_between(x_pos, visit_readmit['Rate'], alpha=0.3, color='#e74c3c')
    axes[idx].set_xlabel(f'Number of Prior {label} (capped at 5)', fontweight='bold')
    axes[idx].set_ylabel('Readmission Rate (%)', fontweight='bold')
    axes[idx].set_title(f'Readmission vs {label}', fontweight='bold', fontsize=11)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xticks(x_pos)
    axes[idx].set_xticklabels([str(int(x)) if x < 5 else '5+' for x in x_pos])

plt.tight_layout()
plt.savefig(f'{output_dir}12_visit_type_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 12_visit_type_breakdown.png")

# Visualization 13: Compound analysis - visits and diagnoses
print("\nCreating compound analysis: Utilization + Diagnosis Complexity...")

diabetic_data['diagnosis_complexity'] = pd.cut(diabetic_data['number_diagnoses'],
                                                bins=[0, 5, 7, 9, 16],
                                                labels=['Low (1-5)', 'Medium (6-7)', 'High (8-9)', 'Very High (10+)'])

compound_analysis = diabetic_data.groupby(['utilization_category', 'diagnosis_complexity'])['readmitted_binary'].mean() * 100
compound_analysis = compound_analysis.unstack()

fig, ax = plt.subplots(figsize=(12, 6))
compound_analysis.plot(kind='bar', ax=ax, colormap='RdYlGn_r', alpha=0.8, edgecolor='black')
ax.set_xlabel('Prior Utilization Category', fontweight='bold')
ax.set_ylabel('Readmission Rate (%)', fontweight='bold')
ax.set_title('Readmission Rate by Utilization and Diagnosis Complexity', fontweight='bold', fontsize=12)
ax.legend(title='Diagnosis Complexity', title_fontsize=10, fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')

plt.tight_layout()
plt.savefig(f'{output_dir}13_compound_utilization_diagnosis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 13_compound_utilization_diagnosis.png")

# ==============================================================================
# SECTION 6: KEY INSIGHTS SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS SUMMARY")
print("="*80)

print("\n1. TARGET VARIABLE (READMISSION):")
print(f"   - Overall readmission rate (<30 days): {diabetic_data['readmitted_binary'].mean()*100:.2f}%")
print(f"   - Total patients: {len(diabetic_data):,}")
print(f"   - Readmitted within 30 days: {diabetic_data['readmitted_binary'].sum():,}")

print("\n2. AGE PATTERNS:")
age_stats = age_readmit['Rate']
print(f"   - Highest readmission rate: {age_stats.idxmax()} ({age_stats.max():.2f}%)")
print(f"   - Lowest readmission rate: {age_stats.idxmin()} ({age_stats.min():.2f}%)")
print(f"   - Mean age: {diabetic_data['age_numeric'].mean():.1f} years")

print("\n3. PRIOR UTILIZATION (CRITICAL FINDING):")
print(f"   - Patients with 0 prior inpatient visits: {inpatient_readmit.loc['0', 'Rate']:.2f}% readmission")
print(f"   - Patients with 5+ prior inpatient visits: {inpatient_readmit.loc['5+', 'Rate']:.2f}% readmission")
print(f"   - Increase factor: {inpatient_readmit.loc['5+', 'Rate'] / inpatient_readmit.loc['0', 'Rate']:.2f}x")

print("\n4. DIAGNOSIS CATEGORIES:")
print(f"   - Highest risk diagnosis: {diag_readmit.index[0]} ({diag_readmit['Rate'].iloc[0]:.2f}%)")
print(f"   - Lowest risk diagnosis: {diag_readmit.index[-1]} ({diag_readmit['Rate'].iloc[-1]:.2f}%)")

print("\n5. STRONGEST PREDICTORS (CORRELATION):")
for feature, corr_val in readmit_corr.head(3).items():
    print(f"   - {feature}: {corr_val:.4f}")

print("\n6. PATIENT COMPLEXITY:")
print(f"   - Mean time in hospital: {diabetic_data['time_in_hospital'].mean():.1f} days")
print(f"   - Mean number of procedures: {diabetic_data['num_procedures'].mean():.1f}")
print(f"   - Mean number of medications: {diabetic_data['num_medications'].mean():.1f}")

print("\n" + "="*80)
print("EDA COMPLETE - All visualizations saved to ./output/data_exploration/")
print("="*80)

# Create a summary dataframe for easy reference
summary_data = {
    'Metric': [
        'Total Patients',
        'Readmitted <30 days',
        'Readmission Rate',
        'Mean Age (years)',
        'Mean Hospital Stay (days)',
        'Mean Medications',
        'Mean Diagnoses'
    ],
    'Value': [
        f"{len(diabetic_data):,}",
        f"{diabetic_data['readmitted_binary'].sum():,}",
        f"{diabetic_data['readmitted_binary'].mean()*100:.2f}%",
        f"{diabetic_data['age_numeric'].mean():.1f}",
        f"{diabetic_data['time_in_hospital'].mean():.2f}",
        f"{diabetic_data['num_medications'].mean():.2f}",
        f"{diabetic_data['number_diagnoses'].mean():.2f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\nDataset Summary:")
print(summary_df.to_string(index=False))

print("\n✓ Script execution completed successfully!")
print(f"✓ Generated 13 comprehensive visualizations")
print(f"✓ All files saved to: ./output/data_exploration/")