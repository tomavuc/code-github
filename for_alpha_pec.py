import json
import os
import time
import requests
import re

import pandas as pd

from Bio import Entrez, SeqIO
from feature_generation import fetch_records

df = pd.read_csv('datasets/mydatabep-uniprot.csv')
stop = 76
organism_names = df.iloc[:stop, 0]
genbank_ids = [gen.strip() for gen in df.iloc[:stop, 1].dropna().tolist()]
pIspG_values = df.iloc[:stop, 2]
uniprot_ids = [uid.strip() for uid in df.iloc[:stop, 3].dropna().tolist()]

for acc, value, org in zip(genbank_ids, pIspG_values, organism_names):
    print(f"Organism: {org}, Accession Number: {acc}, pIspG: {value}")

Entrez.email = 'tomasinho7778@gmail.com'

# For example:
gb_accessions = list(genbank_ids)[-2:]
records = fetch_records(gb_accessions, batch_size=10)

jobs = []
# Sequence of Flavodoxin A, taken from E. Coli as its PEC, can be changed
flavo_seq = "MAITGIFFGSDTGNTENIAKMIQKQLGKDVADVHDIAKSSKEDLEAYDILLLGIPTWYYGEAQCDWDDFFPTLEEIDFNGKLVALFGCGDQEDYAEYFCDALGTIRDIIEPRGATIVGHWPTAGYHFEASKGLADDDHFVGLAIDEDRQPELTAERVEKWVKQISEELHLDEILNA"  

jobs = []
for i, record in enumerate(records):
    rid = getattr(record, "id", None) or f"record_{i}"
    safe_id = re.sub(r"\W+", "_", rid)

    ispG_seq  = str(record.seq).strip()

    if not ispG_seq:
        print(f"Skipping empty IspG at {safe_id}")
        continue

    # 3) build the one job containing both dimer + carrier chains
    job = {
      "name": f"{safe_id}_with_fldA",
      "modelSeeds": [],
      "sequences": [
        { "proteinChain": { "sequence": ispG_seq,  "count": 2 } },
        { "proteinChain": { "sequence": flavo_seq, "count": 1 } },
        { "proteinChain": { "sequence": flavo_seq, "count": 1 } }
      ],
      "dialect": "alphafoldserver",
      "version": 1
    }
    jobs.append(job)

# 4) dump to JSON
with open("batch_ispG_fldA_complex.json", "w") as f:
    json.dump(jobs, f, indent=2)

print(f"Wrote {len(jobs)} complex jobs to batch_ispG_fldA_complex.json")
