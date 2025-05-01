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

_atomic_masses = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06}

# Sequence based features (GenBank)
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
        
        time.sleep(1)
    return all_records

def extract_seq_features(record) -> dict:
    seq = str(record.seq)
    features = {}
    
    features['GenBankID'] = record.id
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

# GO Annotations features
def get_go_annotations(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?fields=go&format=json"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching GO annotations for {uniprot_id}: {response.status_code}")
        return set()
    data = response.json()
    go_terms = set()
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "GO":
            go_terms.add(ref.get("id"))
    return go_terms

def generate_go_feature_matrix(uniprot_ids):
    protein_go = {}
    all_go_terms = set()
    for uid in uniprot_ids:
        go_terms = get_go_annotations(uid)
        protein_go[uid] = go_terms
        all_go_terms.update(go_terms)
        time.sleep(2)
    
    all_go_terms = sorted(list(all_go_terms))
    feature_data = []
    for uid in uniprot_ids:
        features = {"UniProtID": uid}
        for go in all_go_terms:
            features[go] = 1 if go in protein_go.get(uid, set()) else 0
        feature_data.append(features)

    return pd.DataFrame(feature_data)

#Downloading 3D structures

def get_uniprot_crossrefs(uniprot_id):
    uniprot_id = uniprot_id.strip()
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?fields=xref_pdb&format=json"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching cross-references for {uniprot_id}: {response.status_code}")
        print("Response content:", response.text)
        return {"PDB": []}
    
    entry = response.json()
    pdb_refs = []
    for ref in entry.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "PDB":
            pdb_refs.append(ref.get("id"))
    return {"PDB": pdb_refs}

def download_pdb_structure(pdb_id, directory="."):
    pdbl = PDBList()
    file_path = pdbl.retrieve_pdb_file(pdb_id, file_format="mmCif", pdir=directory)
    return file_path

def download_alphafold_structure(uniprot_id, directory="."):
    alphafold_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif"
    response = requests.get(alphafold_url)
    if response.status_code == 200:
        file_name = f"AF-{uniprot_id}-F1-model_v4.cif"
        file_path = os.path.join(directory, file_name)
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        print(f"AlphaFold model not found for {uniprot_id} (status code {response.status_code}).")
        return None

def download_structure_by_uniprot(uniprot_id, directory="."):
    crossrefs = get_uniprot_crossrefs(uniprot_id)
    if crossrefs.get("PDB"):
        pdb_id = crossrefs["PDB"][0]
        print(f"Downloading PDB structure for {uniprot_id} (PDB ID: {pdb_id})")
        return download_pdb_structure(pdb_id, directory)
    else:
        print(f"No PDB structure found for {uniprot_id}, trying AlphaFold...")
        return download_alphafold_structure(uniprot_id, directory)

# Calculating the RMSE between the protein of interest and the E. Coli

def compute_rmse(query_file, ecoli_file):
    cmd.reinitialize()

    query_obj = "query"
    ecoli_obj = "ecoli"
    cmd.load(query_file, query_obj)
    cmd.load(ecoli_file, ecoli_obj)

    rms_tuple = cmd.super(query_obj, ecoli_obj)
    rms_value = rms_tuple[0]
    return rms_value

# Calculating the opening of the binding site

def get_uniprot_binding_sites(uniprot_id): 
    uniprot_id = uniprot_id.strip()
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

def compute_binding_site_opening(cif_path, uniprot_id, chain_id="A"):
    sites = get_uniprot_binding_sites(uniprot_id)
    cys_resnums = sites["CYS"]
    glu_resnums = sites["GLU"]
    if len(cys_resnums) != 3 or len(glu_resnums) != 1:
        raise ValueError(f"Expected 3 CYS and 1 GLU for {uniprot_id}, got {sites}")

    parser = MMCIFParser()
    struct_id = os.path.basename(cif_path).split('.')[0]
    structure = parser.get_structure(struct_id, cif_path)
    model = structure[0]
    chain = model[chain_id]

    def _collect_atoms(resnums):
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
            raise ValueError(f"No atoms found for residues {resnums}")
        masses = np.array(masses)
        coords = np.array(coords)

        com = (masses[:,None] * coords).sum(axis=0) / masses.sum()
        return com

    com_cys = _collect_atoms(cys_resnums)
    com_glu = _collect_atoms(glu_resnums)
    opening = np.linalg.norm(com_cys - com_glu)
    return float(opening)

if __name__ == "__main__":
    gb_accessions = list(genbank_ids)
    batch_size = 1
    records = fetch_records(gb_accessions, batch_size)
    seq_features = [extract_seq_features(record) for record in records]
    df_seq = pd.DataFrame(seq_features)
    print("Sequence Features:")
    print(df_seq.head())

    df_go = generate_go_feature_matrix(uniprot_ids)
    print("GO Feature Matrix:")
    print(df_go.head())

    if os.path.exists("structures"):
        shutil.rmtree("structures")
    os.makedirs("structures", exist_ok=True)
    
    structure_files = {}
    for uid in uniprot_ids:
        file_path = download_structure_by_uniprot(uid, directory="structures")
        structure_files[uid] = file_path
        time.sleep(1)
    
    ecoli_structure_file = "e_coli_model.cif"
    if not os.path.exists(ecoli_structure_file):
        print("Warning: E. coli reference structure not found at", ecoli_structure_file)
    
    rmse_results = {}
    for uid, struct_file in structure_files.items():
        if struct_file is not None and os.path.exists(ecoli_structure_file):
            rmse = compute_rmse(struct_file, ecoli_structure_file)
            rmse_results[uid] = rmse
            print(f"RMSE for {uid}: {rmse}")
        #else:
        #    rmse_results[uid] = None

    opening_results = {}
    for uid, cif_path in structure_files.items():
        if cif_path and os.path.exists(cif_path):
            try:
                opening = compute_binding_site_opening(cif_path, uid, chain_id="A")
            except Exception as e:
                print(f"Could not compute opening for {uid}: {e}")
                opening = None
        else:
            opening = None
        opening_results[uid] = opening
        time.sleep(1)

    df_opening = pd.DataFrame.from_dict(opening_results, orient="index", columns=["binding_site_opening"]).reset_index().rename(columns={"index":"UniProtID"})

    df_rmse = pd.DataFrame.from_dict(rmse_results, orient='index', columns=['RMSE'])
    df_rmse.index.name = "UniProtID"
    df_rmse = df_rmse.reset_index() 

    df_structure = pd.merge(df_go, df_rmse, on="UniProtID", how="outer")
    merged_seq = pd.merge(map, df_seq, on="GenBankID", how="left")

    final_df = pd.merge(merged_seq, df_structure, on="UniProtID", how="outer").merge(df_opening, on="UniProtID", how="left")

    print("Final Feature Matrix:")
    print(final_df.head())
    
    final_df.to_csv("datasets/final_feature_matrix_with_opening.csv")