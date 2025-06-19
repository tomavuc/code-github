import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_feature = 'RMSE'

df = pd.read_csv("all_merged_final.csv")
feature = df.sort_values(input_feature)
print(feature[input_feature])
print(np.mean(feature[input_feature]))

colours = [
    'green' if lab == '+' else
    'red'  if lab == '-' else
    'grey'
    for lab in feature['pIspG']]

plt.figure(figsize=(12, max(10, 0.25 * len(feature))))
plt.barh(feature['id'], feature[input_feature], color=colours, edgecolor='black')
plt.xlabel(f"{input_feature} value")
plt.title(f"Sorted {input_feature}")
plt.tight_layout()
plt.show()