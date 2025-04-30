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

map = pd.DataFrame()
map['GenBankID'] = df.iloc[:stop, 1].astype(str).str.strip()
map['UniProtID'] = df.iloc[:stop, 3].astype(str).str.strip()
map['pIspG'] = df.iloc[:stop, 2]
print("Mapping Table:")
print(map.head())

for acc, value, org in zip(genbank_ids, pIspG_values, organism_names):
    print(f"Organism: {org}, Accession Number: {acc}, pIspG: {value}")

Entrez.email = 'tomasinho7778@gmail.com'

# For example:
gb_accessions = list(genbank_ids)
records = fetch_records(gb_accessions, batch_size=10)
# Here each record contains the protein sequence (accessible via record.seq)

jobs = []

for i, record in enumerate(records):
    # Use record.id if available, otherwise use an index-based fallback.
    record_id = getattr(record, 'id', None) or f"record_{i}"
    # Sanitize the ID: only letters, numbers, and underscores allowed.
    safe_id = re.sub(r'\W+', '_', record_id)
    job_name = f"{safe_id}_dimer"

    protein_seq = str(record.seq).strip()
    if not protein_seq:
        print(f"Skipping record {safe_id} with empty sequence.")
        continue

    # Create the job object following the JSON schema from the README.
    job = {
        "name": job_name,
        "modelSeeds": [],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": protein_seq,
                    "count": 2  # set to 2 for self-dimerization
                }
            }
        ],
        "dialect": "alphafoldserver",
        "version": 1
    }
    jobs.append(job)

# Write the JSON file with all the job definitions
with open("batch_jobs.json", "w") as outfile:
    json.dump(jobs, outfile, indent=2)

print(f"{len(jobs)} job(s) written to 'batch_jobs.json'")
