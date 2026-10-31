import pandas as pd
import numpy as np

dd = pd.read_csv(r'hack-day-lucknow-datasets/p1-clean-cooking/data_dictionary.csv')
print("--- DATA DICTIONARY SAMPLE ---")
for idx, r in dd.head(45).iterrows():
    print(f"{r['column']} | dtype: {r['dtype']} | ex: {r['example']} | 88/99: {r['has_code_88_or_99']}")

df = pd.read_csv(r'hack-day-lucknow-datasets/p1-clean-cooking/train.csv', nrows=1000)
print("\n--- SAMPLE TARGET INFO ---")
print(df['target_solid_fuel'].value_counts())
print("\n--- SAMPLE COLUMNS ---")
print(list(df.columns[:20]))

