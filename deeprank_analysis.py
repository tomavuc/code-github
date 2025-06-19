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

def build_feature_df(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    feat_df = pd.DataFrame({"id": df["id"].apply(extract_accession)})

    # Buried and solvend accessible surface areas
    bsa_arrays  = df["bsa"].apply(parse_array)
    sasa_arrays = df["sasa"].apply(parse_array)

    feat_df["dr_interface_size"] = bsa_arrays.apply(len)
    feat_df["dr_bsa"]  = np.round(bsa_arrays.apply(sum),  2)
    feat_df["dr_sasa"] = np.round(sasa_arrays.apply(sum), 2)

    # One hot encoding of occupancy per residue type
    res_types = [c for c in df.columns if c.startswith("res_type_")]
    for col in res_types:
        arrs = df[col].apply(parse_array)
        feat_df[f"dr_{col}"] = arrs.apply(lambda a: np.mean(a) if a else 0)

    # Polarity group occupancy
    polarity_cols = [c for c in df.columns if c.startswith("polarity_")]
    for col in polarity_cols:
        arrs = df[col].apply(parse_array)
        feat_df[f"dr_{col}"] = arrs.apply(lambda a: np.mean(a) if a else 0)

    # Features for continuous sums 
    for feat in ["res_size", "res_mass", "res_charge", "hb_acceptors", "hb_donors"]:
        arrs = df[feat].apply(parse_array)
        feat_df[f"dr_{feat}"] = np.round(arrs.apply(sum), 3)

    # IRC features
    irc_cols = [c for c in df.columns if c.startswith("irc_")]
    for col in irc_cols:
        arrs = df[col].apply(parse_array)
        feat_df[f"dr_{col}"] = np.round(arrs.apply(sum), 3)

    # 1) split net charge into positive & negative components
    charge_arrays = df["res_charge"].apply(parse_array)
    feat_df["dr_res_charge_positive"] = charge_arrays.apply(lambda a: sum(c for c in a if c > 0))
    feat_df["dr_res_charge_negative"] = charge_arrays.apply(lambda a: sum(c for c in a if c < 0))

    # 2) mean isoelectric point (pI)
    pI_arrays = df["res_pI"].apply(parse_array)
    feat_df["dr_res_pI"] = np.round(pI_arrays.apply(lambda a: np.mean(a) if a else 0), 2)

    # 3) H-bond donor / acceptor ratio
    d_sum = feat_df["dr_hb_donors"]
    a_sum = feat_df["dr_hb_acceptors"]
    feat_df["dr_hb_donor_acceptor_ratio"] = np.where(a_sum > 0, d_sum / a_sum, 0)

    # 4) salt-bridge enrichment  (neg-pos contacts divided by total)
    neg_pos = feat_df.get("dr_irc_negative_positive", 0)
    pos_neg = feat_df.get("dr_irc_positive_negative", 0)
    total   = feat_df["dr_irc_total"]
    feat_df["dr_salt_bridges"] = np.where(total > 0, (neg_pos + pos_neg) / total, 0)

    same_chain = df["same_chain"].apply(parse_array)
    electro    = df["electrostatic"].apply(parse_array)
    vdw        = df["vanderwaals"].apply(parse_array)

    # edge energies
    feat_df["dr_electrostatic_inter"] = [
        np.round(sum(e for e, sc in zip(elec, scs) if sc == 0), 2)
        for elec, scs in zip(electro, same_chain)
    ]
    feat_df["dr_vanderwaals_inter"] = [
        np.round(sum(v for v, sc in zip(vd, scs) if sc == 0), 2)
        for vd, scs in zip(vdw, same_chain)
    ]

    return feat_df

new_df_C = build_feature_df("merged_features_C_residue_full_v2.csv")
new_df_D = build_feature_df("merged_features_D_residue_full.csv")

avg_df = (new_df_C.set_index("id").add(new_df_D.set_index("id")).div(2).reset_index())

print("\nAveraged feature matrix:")
print(avg_df.head())
#avg_df.to_csv("deeprank_features_v2.csv", index=False)