import pandas as pd
import numpy as np

train_path = r'hack-day-lucknow-datasets/p1-clean-cooking/train.csv'
df = pd.read_csv(train_path)

# Drop ID and target for correlation check
target = df['target_solid_fuel']
num_cols = df.select_dtypes(include=[np.number]).columns.drop(['household_id', 'target_solid_fuel'], errors='ignore')

corrs = df[num_cols].apply(lambda x: x.corr(target)).dropna()
top_pos = corrs.sort_values(ascending=False).head(15)
top_neg = corrs.sort_values().head(15)

print("--- TOP POSITIVE CORRELATIONS WITH SOLID FUEL (Higher = More likely Chulha) ---")
print(top_pos)

print("\n--- TOP NEGATIVE CORRELATIONS WITH SOLID FUEL (Higher = More likely Clean Fuel) ---")
print(top_neg)

