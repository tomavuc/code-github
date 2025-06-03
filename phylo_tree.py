from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

alignment_path = "ispG_full.aln-clustalw"
alignment = AlignIO.read(alignment_path, "clustal")

check = pd.read_csv('ispG_MSA_features_full.csv')
print(check['pid_total'].mean(), check['pid_total'].std())

calc = DistanceCalculator("blosum80") # protein model
dm = calc.get_distance(alignment)

df_tri = pd.DataFrame(dm.matrix, index=dm.names, columns=dm.names)
df = df_tri.where(~df_tri.isna(), df_tri.T) 
print(df.head)
ref_substring = "AAC75568.1"
dist_series = df[ref_substring]

csv_path = "ispG_distances_to_AAC75568.1.csv"
dist_series.to_csv(csv_path, header=["Distance"], index_label= 'id')

plt.figure(figsize=(8, max(4, 0.25 * len(dist_series))))
dist_series.sort_values().plot(kind="barh")
plt.xlabel("Evolutionary distance (Blosum80)")
plt.title(f"Distance to E. coli reference ({ref_substring})")
plt.tight_layout()
plt.show()


