# ----------------- Imports -----------------
import os
import numpy as np
import pandas as pd
import requests
from Bio.PDB import MMCIFParser

# ----------------- Constants -----------------
# atomic masses for COM
_atomic_masses = {"H":1.008,"C":12.011,"N":14.007,"O":15.999,"S":32.06,"Se":78.96,"P":30.973}

# fixed PEC interface positions on flavodoxin (E. coli)
FLDA_POSITIONS = (
    list(range(10,16))   # 10-15
  + list(range(56,61))   # 56-60
  + [90]                 # 90
  + list(range(94,100))  # 94-99
  + [147]                # 147
)

# ----------------- Helper functions -----------------
def get_uniprot_binding_sites(uniprot_id, id_map, use_map):
    old = uniprot_id
    if use_map == True:
        matches = id_map.loc[id_map['GenBankID'] == str(uniprot_id), 'UniProtID']
        uniprot_id = matches.iloc[0]  
    uniprot_id = uniprot_id.strip()
    print(old, uniprot_id)
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url)
    response.raise_for_status()
    entry = response.json()
    positions = []


    for feat in entry.get("features", []):
        if feat.get("type").upper() == "BINDING SITE":
            loc = feat.get("location", {})
            value = loc.get("start", {}).get("value")
            if value is None:
                continue
            try:
                positions.append(int(value))
            except ValueError:
                continue

    if len(positions) < 4:
        raise RuntimeError(f"UniProt {uniprot_id} has fewer than 4 binding‐site annotations: found {len(positions)}")
    
    positions_sorted = sorted(positions)
    cys_sites = positions_sorted[:3]
    glu_site  = positions_sorted[-1]

    return {"CYS": cys_sites, "GLU": [glu_site]}

def compute_com(chain, resnums):
    masses = []
    coords = []
    for resnum in resnums:
        res = chain[(' ', resnum, ' ')]
        for atom in res:
            elem = atom.element.strip()
            m = _atomic_masses.get(elem)
            if m is None:
                continue
            masses.append(m)
            coords.append(atom.coord)
    if not masses:
        raise ValueError(f"No atoms found for residues {resnums} in chain {chain}")
    masses = np.array(masses)
    coords = np.array(coords)
    com = (masses[:,None] * coords).sum(axis=0) / masses.sum()
    return com

def compute_cluster_flda_distance(cif_path, gb_id,
                                  id_map=None,
                                  cluster_chains=("A","B"),
                                  flda_chains=("C","D")):
    """Return (dist_chainA, dist_chainB)."""
    parser = MMCIFParser()
    model  = parser.get_structure("complex", cif_path)[0]

    # COM of Fe-S binding site (4 residues) for each IspG chain
    sites = get_uniprot_binding_sites(gb_id, id_map, use_map=True)
    cluster_res = sites["CYS"] + sites["GLU"]
    ispG_com = {c: compute_com(model[c], cluster_res) for c in cluster_chains}

    # COM of PEC interface for each flavodoxin chain
    flda_com = {c: compute_com(model[c], FLDA_POSITIONS) for c in flda_chains}

    # distance matrix and greedy pairing
    dmat = {(i,f): np.linalg.norm(ispG_com[i]-flda_com[f])
            for i in cluster_chains for f in flda_chains}
    
    (i1,f1), d1 = min(dmat.items(), key=lambda kv: kv[1])
    (i2,f2), d2 = min({k:v for k,v in dmat.items()
                       if k[0]!=i1 and k[1]!=f1}.items(),
                      key=lambda kv: kv[1])
    return (d1,d2) if cluster_chains[0]==i1 else (d2,d1)

def derive_gb_id(model_key):
    """fold_wp_00002314_1  ->  WP_00002314.1   (or AAC...)"""
    parts = model_key.split("_")
    if len(parts)>=4:
        accession = f"{parts[1]}_{parts[2]}"; version = parts[3]
    else:
        accession, version = parts[1], parts[2]
    return f"{accession.upper()}.{version}"

# ----------------- Test harness -----------------
df = pd.read_csv('datasets/mydatabep-uniprot.csv')
complex_dir = "dimers_with_pec"
stop = 78
mapa = pd.DataFrame()
mapa['GenBankID'] = df.iloc[:stop, 1].astype(str).str.strip()
mapa['UniProtID'] = df.iloc[:stop, 3].astype(str).str.strip()
mapa['pIspG'] = df.iloc[:stop, 2]

records = []
for fn in os.listdir(complex_dir):
    if fn.endswith(".cif") and "_with_flda_" in fn:
        key   = fn.split("_with_flda")[0]
        gb_id = derive_gb_id(key)
        cif   = os.path.join(complex_dir, fn)
        try:
            dA, dB = compute_cluster_flda_distance(cif, gb_id, id_map=mapa,
                                                   cluster_chains=("A","B"),
                                                   flda_chains=("C","D"))
            records.append({"GenBankID": gb_id,
                            "CL_FLDA_A": dA,
                            "CL_FLDA_B": dB})
        except Exception as e:
            print(f"{gb_id}: {e}")

df_flda = pd.DataFrame(records)
print(df_flda.head())
