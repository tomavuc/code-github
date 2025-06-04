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
ref_id = "AAC75568.1"
dist_series = df[ref_id]

df_labels = pd.read_csv('datasets/mydatabep-uniprot.csv')
map = pd.DataFrame()
map['id'] = df_labels.iloc[:, 1].astype(str).str.strip()
map['label'] = df_labels.iloc[:, 2]
label_df = map.replace({'+/-': '+'})

merged = (
    dist_series.rename("phylo_dist").reset_index()
      .rename(columns={"index": "id"})
      .merge(label_df, on="id", how="left"))

merged_sorted = merged.sort_values("phylo_dist")

colours = [
    'blue' if lab == '+' else
    'red'  if lab == '-' else
    'grey'
    for lab in merged_sorted['label']]

plt.figure(figsize=(8, max(4, 0.25 * len(merged_sorted))))
plt.barh(
    merged_sorted['id'], 
    merged_sorted['phylo_dist'],  
    color=colours,                
    edgecolor='black'
)
plt.xlabel("Evolutionary distance (Blosum80)")
plt.title("Distance to E. coli reference (AAC75568.1)")
plt.tight_layout()
plt.show()