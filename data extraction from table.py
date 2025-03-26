import pandas as pd
import numpy as np
import requests
from Bio import Entrez
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight
from collections import Counter
import time
import requests

import matplotlib.pyplot as plt

df = pd.read_csv('mydatabep.csv')
accession_numbers = df.iloc[:, 1]  
pIspG_values = df.iloc[:, 2]       

for acc, value in zip(accession_numbers, pIspG_values):
    print(f"Accession Number: {acc}, pIspG: {value}")

Entrez.email = 'tomasinho7778@gmail.com'

def fetch_records(accessions: list[str], batch_size: int):
    all_records = []
    batches = [accessions[i : i + batch_size] for i in range(0, len(accessions), batch_size)]
    for batch in batches:
        try:
            with Entrez.efetch(db = 'protein', id = ','.join(batch), rettype = 'gb', retmode = 'text') as handle:
                records = list(SeqIO.parse(handle, "genbank"))
                all_records.extend(records)
        except Exception as e:
            print(f"Failed for batch {batch}: {e}")
        
        time.sleep(3)
    return all_records

def extract_seq_features(record) -> dict:
    seq = str(record.seq)
    features = {}
    
    features['accession'] = record.id
    features['length'] = len(seq)
    
    # Amino acid properties
    aa_counts = Counter(seq)
    total = sum(aa_counts.values())
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        features[f'aa_{aa}'] = aa_counts.get(aa, 0) / total * 100
    
    #Charged amino acids are: 
    features['charged_aa'] = (aa_counts.get('D',0) + aa_counts.get('E',0) + 
                             aa_counts.get('K',0) + aa_counts.get('R',0)) / total * 100
    
    #Hydrophobic amino acids are:
    features['hydrophobic_aa'] = (aa_counts.get('A',0) + aa_counts.get('V',0) + 
                                 aa_counts.get('I',0) + aa_counts.get('L',0)) / total * 100
    
    #Cysteine and histidine coordinate iron-sulphur protein clusters
    features['cysteine_content'] = aa_counts.get('C', 0) / total * 100
    features['histidine_content'] = aa_counts.get('H', 0) / total * 100
    
    # Phyical/Chemical properties
    try:
        analysis = ProteinAnalysis(seq)
        features['molecular_weight'] = analysis.molecular_weight()
        features['isoelectric_point'] = analysis.isoelectric_point()
        features['aromaticity'] = analysis.aromaticity()
        features['instability_index'] = analysis.instability_index()
    except Exception as e:
        print(f"Error in protein analysis for {record.id}: {e}")
    
    # GenBank annotations
    features['organism'] = record.annotations.get('organism', None)
    
    return features

accessions = list(accession_numbers)
batch_size = 5
records = fetch_records(accessions, batch_size)

all_features = [extract_seq_features(record) for record in records]

df = pd.DataFrame(all_features)
print(df.head())
