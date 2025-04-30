import os
import time
import requests
import re
import shutil

import pandas as pd
import numpy as np

from pymol import cmd
from Bio import Entrez, SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import PDBList, PDBParser, Superimposer, MMCIFParser

from feature_generation import get_uniprot_binding_sites, compute_binding_site_opening

df = pd.read_csv('datasets/mydatabep-uniprot.csv')
stop = 76
organism_names = df.iloc[:stop, 0]
genbank_ids = [gen.strip() for gen in df.iloc[:stop, 1].dropna().tolist()]
pIspG_values = df.iloc[:stop, 2]
uniprot_ids = [uid.strip() for uid in df.iloc[:stop, 3].dropna().tolist()]

map = pd.DataFrame()
map['GenBankID'] = df.iloc[:stop, 1].astype(str).str.strip()
map['UniProtID'] = df.iloc[:stop, 3].astype(str).str.strip()
map['pIspG'] = df.iloc[:stop, 2]

Entrez.email = 'tomasinho7778@gmail.com'

_atomic_masses = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
}

# load just your UniProt IDs
df = pd.read_csv('datasets/mydatabep-uniprot.csv')
uniprot_ids = [uid.strip() for uid in df.iloc[:76, 3].dropna().tolist()]

url = "https://rest.uniprot.org/uniprotkb/O67496.json"
response = requests.get(url)
response.raise_for_status()
entry = response.json()

# f = entry.get("features", [])
# #print(f)
# for feat in f:
#     if feat.get("type", {}).upper() == 'BINDING SITE':
#         loc = feat.get("location", {})
#         print(loc)
#         start = loc.get("start", {}).get("value")
#         print(start)
#         #val = start.get("value")
#         #print(val)


if __name__ == "__main__":
    # 1) Test the binding‐site fetcher alone
    print("=== Testing get_uniprot_binding_sites ===")
    for uid in uniprot_ids[:10]:            # just try the first 10 for now
        try:
            sites = get_uniprot_binding_sites(uid)
            print(f"{uid:12s} → CYS: {sites['CYS']}, GLU: {sites['GLU']}")
        except Exception as e:
            print(f"{uid:12s} → ERROR: {e}")
        time.sleep(1)

    # 2) Pick one UniProt that *did* return sites and test the COM/distance
    good = None
    for uid in uniprot_ids:
        try:
            s = get_uniprot_binding_sites(uid)
            if len(s['CYS']) == 3 and len(s['GLU']) == 1:
                good = uid
                break
        except:
            continue

    if not good:
        print("No entry with 3 CYS + 1 GLU found in those first 76 IDs.")
        exit(1)

    print(f"\n=== Testing compute_binding_site_opening on {good} ===")
    cif_path = f"structures/AF-{good}-F1-model_v4.cif"
    if not os.path.exists(cif_path):
        print("  → CIF not found at", cif_path)
        exit(1)

    try:
        opening = compute_binding_site_opening(cif_path, good, chain_id="A")
        print(f"  Opening distance for {good}: {opening:.2f} Å")
    except Exception as e:
        print(f"  ERROR computing opening for {good}: {e}")