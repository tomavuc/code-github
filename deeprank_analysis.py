import numpy as np
import pandas as pd
import re

def extract_accession(full_id: str) -> str:
    m = re.search(r"fold_([a-z0-9_]+?)_with", full_id)
    if not m:
        raise ValueError(f"Could not parse accession from {full_id}")
    raw = m.group(1)               # e.g. 'wp_011137085_1'  or 'aac07467_1'
    prefix, version = raw.rsplit("_", 1)   # split on last underscore
    acc = f"{prefix.upper()}.{version}"
    return acc

#Explain the parsing
def parse_array(cell):
    if isinstance(cell, str):
        s = cell.strip("[]")
        parts = []
        for line in s.splitlines():
            parts.extend(line.split())
        toks = [tok for tok in parts if not tok.startswith("...")]
        result = []
        for tok in toks:
            try:
                tok_clean = tok.rstrip(',')
                if ('.' in tok_clean) or ('e' in tok_clean.lower()):
                    result.append(float(tok_clean))
                else:
                    result.append(int(tok_clean))
            except:
                continue
        return result
    return []

file = "merged_features_C_residue_new.csv"
df = pd.read_csv(file)
new_df = pd.DataFrame({"id": df["id"].apply(extract_accession)})

bsa_arrays = df["bsa"].apply(parse_array)
sasa_arrays = df["sasa"].apply(parse_array)

new_df["interface_size"] = bsa_arrays.apply(len)
new_df["bsa"] = np.round(bsa_arrays.apply(sum), 2)
new_df["sasa"] = np.round(sasa_arrays.apply(sum), 2)

# res_type occupancy
res_type_cols = [r for r in df.columns if r.startswith("res_type_")]
for r in res_type_cols:
    arrs = df[r].apply(parse_array)
    new_df[f"{r}_pct"] = arrs.apply(lambda a: sum(a)/len(a) if a else 0)

# polarity occupancy
polarity_cols = [p for p in df.columns if p.startswith("polarity_")]
for p in polarity_cols:
    arrs = df[p].apply(parse_array)
    new_df[f"{p}_pct"] = arrs.apply(lambda a: sum(a)/len(a) if a else 0)

# continuous sums (without bsa and sasa)
for feat in ["res_size", "res_mass", "res_charge", "hb_acceptors", "hb_donors"]:
    arrs = df[feat].apply(parse_array)
    new_df[f"{feat}_sum"] = np.round(arrs.apply(sum), 3)

# IRC sums
irc_cols = [c for c in df.columns if c.startswith("irc_")]
for c in irc_cols:
    arrs = df[c].apply(parse_array)
    new_df[f"{c}_sum"] = np.round(arrs.apply(sum), 3)

#EXTRA FEATURES
# 1) split net charge into positive & negative components
charge_arrays = df["res_charge"].apply(parse_array)
new_df["res_charge_positive_sum"] = charge_arrays.apply(lambda a: sum(c for c in a if c > 0))
new_df["res_charge_negative_sum"] = charge_arrays.apply(lambda a: sum(c for c in a if c < 0))

# 2) mean isoelectric point (pI)
pI_arrays = df["res_pI"].apply(parse_array)
new_df["res_pI_mean"] = np.round(pI_arrays.apply(lambda a: np.mean(a) if a else 0), 2)

# 3) H-bond donor / acceptor ratio
d_sum = new_df["hb_donors_sum"]
a_sum = new_df["hb_acceptors_sum"]
new_df["hb_donor_acceptor_ratio"] = np.where(a_sum > 0, d_sum / a_sum, 0)

# 4) salt-bridge enrichment  (neg-pos contacts divided by total)
neg_pos = new_df.get("irc_negative_positive_sum", 0)
pos_neg = new_df.get("irc_positive_negative_sum", 0)  # depends on column naming
total   = new_df["irc_total_sum"]
new_df["salt_bridges"] = np.where(total > 0, (neg_pos + pos_neg) / total, 0)

# edge energies
same_chain = df["same_chain"].apply(parse_array)
electro = df["electrostatic"].apply(parse_array)
vdw = df["vanderwaals"].apply(parse_array)

new_df["electrostatic_inter_sum"] = [
    np.round(sum(e for e, sc in zip(elec, scs) if sc == 0), 2)
    for elec, scs in zip(electro, same_chain)]

new_df["vanderwaals_inter_sum"] = [
    np.round(sum(v for v, sc in zip(vd, scs) if sc == 0), 2)
    for vd, scs in zip(vdw, same_chain)]

print(new_df.head)
new_df.to_csv("deeprank_features.csv", index=False)