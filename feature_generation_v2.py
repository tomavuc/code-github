import os
import time
import requests
import gzip

import pandas as pd
import numpy as np

from pymol import cmd
from collections import Counter
from Bio import Entrez, SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import MMCIFParser

_aminoacids_key = {
        "CYS": "C",
        "ASP": "D",
        "SER": "S",
        "GLN": "Q",
        "LYS": "K",
        "ILE": "I",
        "PRO": "P",
        "THR": "T",
        "PHE": "F",
        "ASN": "N",
        "GLY": "G",
        "HIS": "H",
        "LEU": "L",
        "ARG": "R",
        "TRP": "W",
        "ALA": "A",
        "VAL": "V",
        "GLU": "E",
        "TYR": "Y",
        "MET": "M"}

_atomic_masses = {
        "H": 1.008,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "S": 32.06,
        "Se": 78.96,
        "P": 30.973}
    
# binding site, as indicated in UniProt
FLDA_POSITIONS = (list(range(10,16))+ list(range(56,61)) + [90] + list(range(94,100)) + [147])

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

def save_sequences_fasta(records: list[SeqIO.SeqRecord],fasta_path: str,compress: bool = False,) -> str:
    if not fasta_path.lower().endswith(".fasta"):
        fasta_path += ".fasta"
    if compress and not fasta_path.lower().endswith(".gz"):
        fasta_path += ".gz"                                                                                                                                                                                 

    # open the right handle
    open_fn = gzip.open if compress else open
    mode    = "wt"          # text mode, always
    with open_fn(fasta_path, mode) as handle:
        # Build cleaner headers so everything is on one line
        for rec in records:
            acc   = rec.id
            descr = rec.annotations["organism"]     # full GenBank description
            header = f"{acc} {descr.split(' ', 1)[1]}" if ' ' in descr else acc
            handle.write(f">{header}\n")
            handle.write(str(rec.seq) + "\n")

    print(f"Wrote {len(records)} sequences to {os.path.abspath(fasta_path)}")
    return os.path.abspath(fasta_path)

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
        time.sleep(1)
    
    all_go_terms = sorted(list(all_go_terms))
    feature_data = []
    for uid in uniprot_ids:
        features = {"UniProtID": uid}
        for go in all_go_terms:
            features[go] = 1 if go in protein_go.get(uid, set()) else 0
        feature_data.append(features)

    return pd.DataFrame(feature_data)

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

def get_uniprot_binding_sites(uniprot_id, id_map, use_map):
    fallback = {
        "S6G5D3": {"CYS": [294, 297, 340], "GLU": [347]},
        "W7JBQ5": {"CYS": [731, 734, 765], "GLU": [772]}}
    
    old = uniprot_id
    if use_map == True:
        matches = id_map.loc[id_map['GenBankID'] == str(uniprot_id), 'UniProtID']
        uniprot_id = matches.iloc[0]  
    uniprot_id = uniprot_id.strip()
    #print(old, uniprot_id)

    try:
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
    
    except Exception:
        positions = []

    if len(positions) < 4 and uniprot_id in fallback:
        print(f"UniProt {uniprot_id} has been found in the fallback")
        return fallback[uniprot_id]
    
    elif len(positions) < 4 and uniprot_id not in fallback:
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

def compute_binding_site_opening(cif_path, gb_id, threshold = 1.0, chain_ids = ("A", "B")):
    parser = MMCIFParser()
    struct_id = os.path.basename(cif_path).split('.')[0]
    structure = parser.get_structure(struct_id, cif_path)
    model = structure[0]
    distances = []

    sites = get_uniprot_binding_sites(gb_id, id_map = map, use_map= True)
    for chain_id in chain_ids:
        chain = model[chain_id]
        com_cys = compute_com(chain, sites['CYS'])
        com_glu = compute_com(chain, sites['GLU'])
        if len(sites['CYS']) != 3 or len(sites['GLU']) != 1:
            raise ValueError(f"Expected 3 CYS and 1 GLU for {gb_id}, got {sites}")
        distances.append(np.linalg.norm(com_cys - com_glu))
    
    avg_opening = float(np.mean(distances))
    if abs(distances[0] - distances[1]) > threshold:
        print(f"Warning: binding-site openings for {gb_id} differ by {abs(distances[0] - distances[1]):.3f} Å")
    return avg_opening

def compute_cluster_flda_distance(cif_path, gb_id, id_map=None, cluster_chains=("A","B"), flda_chains=("C","D")):
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

def compute_sphere_features(cif_path: str, gb_id: str, radius: float = 10.0, chain_ids = ("A", "B"), prefix = f"Sphere10", key = _aminoacids_key):
    parser = MMCIFParser()
    struct_id = os.path.basename(cif_path).split('.')[0]
    structure = parser.get_structure(struct_id, cif_path)
    model = structure[0]

    sites = get_uniprot_binding_sites(gb_id, id_map=map, use_map=True)
    cluster_res = sites["CYS"] + sites["GLU"]
    per_chain_dicts = []
    
    for chain_id in chain_ids:
        chain  = model[chain_id]
        centre = compute_com(chain, cluster_res)
        selected = []
        for res in chain:
            if res.id[0] != ' ':
                continue
            for atom in res:
                if np.linalg.norm(atom.coord - centre) <= radius:
                    selected.append(res)
                    break

        if not selected:
            per_chain_dicts.append({f"{prefix}_EMPTY": 1})
            continue
        
        seq = "".join(key[res.resname] for res in selected)
        print(seq)
        record = SeqRecord(Seq(seq), id=f"{chain_id}_sphere")
        print(record)

        feats = extract_seq_features(record)
        for k in ("isoelectric_point", "aromaticity", "instability_index", "organism", "GenBankID"):
            feats.pop(k, None)

        per_chain_dicts.append({f"{prefix}_{k}": v for k, v in feats.items()})

    # 4. average across chains
    df_tmp = pd.DataFrame(per_chain_dicts).fillna(0)
    mean_feats = df_tmp.mean(axis=0).to_dict()
    return mean_feats

def load_dimers(folder):
    structures = {}
    for fname in os.listdir(folder):
        if fname.endswith('.pdb') or fname.endswith('.cif'):
            if '_dimer_' in fname:
                key = fname.split('_dimer')[0]
            elif '_with_flda_' in fname:
                key = fname.split('_with_flda')[0]
            
            parts = key.split('_')
            # Hard coded so it sees parts of WP_num.1 and AACnum.1 as the same
            if len(parts) >= 4:
                accession = f"{parts[1]}_{parts[2]}"
                version = parts[3]
            else:
                accession = parts[1]
                version = parts[2]
            gb_id = f"{accession.upper()}.{version}"
            structures[key] = gb_id

    return structures

if __name__ == "__main__":
    df = pd.read_csv('datasets/mydatabep-uniprot.csv')
    stop = 78
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

    gb_accessions = list(genbank_ids)
    batch_size = 1
    records = fetch_records(gb_accessions, batch_size)
    save_sequences_fasta(records, fasta_path="ispG_fullset.fasta", compress=False)
    seq_features = [extract_seq_features(record) for record in records]
    df_seq = pd.DataFrame(seq_features)
    df_seq = pd.merge(map, df_seq, on = "GenBankID", how = "left")
    print("Sequence Features:")
    print(df_seq.head())

    df_go = generate_go_feature_matrix(uniprot_ids)
    print("GO Feature Matrix:")
    print(df_go.head())
    
    # Structure features from dimers
    dimer_dir = "dimers"
    dimers = load_dimers(dimer_dir)
    ref_structure_file = "e_coli_dimer.cif"
    if not os.path.exists(ref_structure_file):
        print("Warning: Reference structure not found at", ref_structure_file)
    
    rmse_results = {}
    opening_results = {}
    sphere_results = {}
    for model_key, gb_id in dimers.items():
        cif_path = os.path.join(dimer_dir, f"{model_key}_dimer_model_0.cif")
        if os.path.exists(cif_path) and os.path.exists(ref_structure_file):
            rmse_results[gb_id] = compute_rmse(cif_path, ref_structure_file)
            try:
                opening_results[gb_id] = compute_binding_site_opening(cif_path, gb_id)
            except Exception as e:
                print(f"Could not compute opening for {gb_id}: {e}")
                opening_results[gb_id] = None
            try:
                sphere_results[gb_id] = compute_sphere_features(cif_path, gb_id, radius=10.0)
            except Exception as e:
                print(f"Sphere-feature extraction failed for {gb_id}: {e}")
                sphere_results[gb_id] = {}
        else:
            rmse_results[gb_id] = None
            opening_results[gb_id] = None
            sphere_results[gb_id] = None

    df_rmse = pd.DataFrame.from_dict(rmse_results, orient='index', columns=['RMSE'])
    df_rmse = df_rmse.reset_index().rename(columns={'index':'GenBankID'})

    df_opening = pd.DataFrame.from_dict(opening_results, orient='index', columns=['binding_site_opening'])
    df_opening = df_opening.reset_index().rename(columns={'index':'GenBankID'})

    df_sphere = pd.DataFrame.from_dict(sphere_results, orient='index')
    df_sphere = df_sphere.reset_index().rename(columns={'index':'GenBankID'})

    # Distance of Fe-S site to binding site of fldA
    complex_dir = 'dimers_with_pec'
    complexes = load_dimers(complex_dir)
    flda_distA, flda_distB = {}, {}
    for model_key, gb_id in complexes.items():
        cif_path = os.path.join(complex_dir, f"{model_key}_with_flda_model_0.cif")
        try:
            dA, dB = compute_cluster_flda_distance(cif_path, gb_id, map)
            flda_distA[gb_id], flda_distB[gb_id] = dA, dB
        except Exception as e:
            print(f"cluster‑FLDA distance failed for {gb_id}: {e}")
            flda_distA[gb_id] = flda_distB[gb_id] = None
    
    df_fldaA = pd.DataFrame.from_dict(flda_distA, orient='index', columns=['CL_FLDA_A']).reset_index().rename(columns={'index':'GenBankID'})
    df_fldaB = pd.DataFrame.from_dict(flda_distB, orient='index', columns=['CL_FLDA_B']).reset_index().rename(columns={'index':'GenBankID'})

    merged_seq = pd.merge(df_seq, df_go, on = 'UniProtID', how='left')
    merged_struct = pd.merge(df_rmse, df_opening, on='GenBankID', how='outer')
    before_df = pd.merge(merged_seq, merged_struct, on='GenBankID', how='left')
    distance_df = pd.merge(df_fldaA, df_fldaB, on='GenBankID', how='left')
    final_df = pd.merge(before_df, distance_df, on='GenBankID', how='left')
    final_df = pd.merge(final_df, df_sphere, on = 'GenBankID', how = 'left')

    print("Final Feature Matrix:")
    print(final_df.head())
    os.makedirs('datasets', exist_ok=True)
    final_df.to_csv("datasets/final_feature_matrix_v3.csv", index=False)