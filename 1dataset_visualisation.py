"""
Data Quality Visualization for Diabetic Readmission Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output/data_profile', exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('./dataset/diabetic_data.csv')
print(f"Dataset shape: {df.shape}\n")

sns.set_style("whitegrid")

# MISSING VALUES
print("="*60)
print("MISSING VALUES")
print("="*60)
df_clean = df.replace('?', np.nan)
missing = pd.DataFrame({
    'Column': df_clean.columns,
    'Missing_Count': df_clean.isnull().sum(),
    'Missing_Pct': (df_clean.isnull().sum() / len(df_clean)) * 100
}).sort_values('Missing_Pct', ascending=False)
missing = missing[missing['Missing_Count'] > 0]
print(f"Columns with missing data: {len(missing)}")
print(missing.head(10).to_string(index=False))

# DUPLICATES
print("\n" + "="*60)
print("DUPLICATES")
print("="*60)
exact_dupes = df.duplicated().sum()
dupe_encounters = df['encounter_id'].duplicated().sum()
multi_patients = len(df['patient_nbr'].value_counts()[df['patient_nbr'].value_counts() > 1])
print(f"Exact duplicates: {exact_dupes}")
print(f"Duplicate encounters: {dupe_encounters}")
print(f"Multi-encounter patients: {multi_patients}")

# OUTLIERS
print("\n" + "="*60)
print("OUTLIERS")
print("="*60)
numerical_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                  'num_medications', 'number_diagnoses']
outlier_stats = []
for col in numerical_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    outlier_stats.append({'Feature': col, 'Outliers': len(outliers),
                          'Outlier_Pct': (len(outliers)/len(df))*100})
outlier_df = pd.DataFrame(outlier_stats)
print(outlier_df.to_string(index=False))

# CLASS BALANCE
print("\n" + "="*60)
print("CLASS BALANCE")
print("="*60)
readmit = df['readmitted'].value_counts()
print(f"\nReadmission Distribution:\n{readmit}")
print(f"\nPercentages:\n{(readmit/len(df)*100).round(2)}")

# VISUALIZATIONS
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS...")
print("="*60)

# 1. Missing Data Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
top_missing = missing.head(10)
colors = ['#d62728' if x>50 else '#ff7f0e' if x>20 else '#2ca02c' for x in top_missing['Missing_Pct']]
ax.barh(top_missing['Column'], top_missing['Missing_Pct'], color=colors, edgecolor='black')
ax.set_xlabel('Missing Percentage (%)', fontweight='bold', fontsize=12)
ax.set_title('Missing Data by Feature', fontweight='bold', fontsize=14)
ax.axvline(50, color='red', linestyle='--', alpha=0.7, label='50%')
ax.axvline(20, color='orange', linestyle='--', alpha=0.7, label='20%')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/data_profile/missing_data_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output/data_profile/missing_data_bar.png")

# 2. Missing Data Heatmap
fig, ax = plt.subplots(figsize=(12, 6))
matrix = df_clean[top_missing['Column'].head(8)].isnull().astype(int).sample(500, random_state=42)
sns.heatmap(matrix.T, cmap='YlOrRd', ax=ax, yticklabels=True, xticklabels=False,
            cbar_kws={'label': 'Missing (1) vs Present (0)'})
ax.set_title('Missing Data Pattern (Sample)', fontweight='bold', fontsize=14)
ax.set_xlabel('Patient Records', fontweight='bold', fontsize=12)
ax.set_ylabel('Features', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('output/data_profile/missing_data_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output/data_profile/missing_data_heatmap.png")

# 3. Duplicate Analysis
fig, ax = plt.subplots(figsize=(10, 6))
dupe_data = {'Exact Duplicates': exact_dupes, 'Duplicate Encounters': dupe_encounters,
             'Multi-encounter Patients': multi_patients}
bars = ax.bar(dupe_data.keys(), dupe_data.values(),
              color=['#d62728', '#ff7f0e', '#1f77b4'], edgecolor='black', width=0.6)
ax.set_ylabel('Count', fontweight='bold', fontsize=12)
ax.set_title('Duplicate Records Analysis', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2., h, f'{int(h):,}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('output/data_profile/duplicates.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output/data_profile/duplicates.png")

# 4. Outlier Detection
fig, ax = plt.subplots(figsize=(10, 6))
colors_out = ['#d62728' if x>10 else '#ff7f0e' if x>5 else '#2ca02c' for x in outlier_df['Outlier_Pct']]
ax.barh(outlier_df['Feature'], outlier_df['Outlier_Pct'], color=colors_out, edgecolor='black')
ax.set_xlabel('Outlier Percentage (%)', fontweight='bold', fontsize=12)
ax.set_title('Outlier Detection (IQR Method)', fontweight='bold', fontsize=14)
ax.axvline(10, color='red', linestyle='--', alpha=0.7, label='10%')
ax.axvline(5, color='orange', linestyle='--', alpha=0.7, label='5%')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/data_profile/outliers.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output/data_profile/outliers.png")

# 5. Class Balance
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(readmit.index, readmit.values, color=['#2ca02c', '#d62728'], edgecolor='black', width=0.5)
ax.set_xlabel('Readmission', fontweight='bold', fontsize=12)
ax.set_ylabel('Count', fontweight='bold', fontsize=12)
ax.set_title('Class Balance (YES = <30 or >30 days)', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    h = bar.get_height()
    pct = (h/len(df))*100
    ax.text(bar.get_x()+bar.get_width()/2., h, f'{int(h):,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('output/data_profile/class_balance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output/data_profile/class_balance.png")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)