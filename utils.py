import pandas as pd
import os
from Bio.PDB import MMCIFParser, PDBIO

def balance_and_split_data(input_path, random_state=42, ratio=1.5):
    df = pd.read_csv(input_path)
    df['pIspG'] = df['pIspG'].replace({'+/-': '+'})

    positive_df = df[df['pIspG'] == '+']
    negative_df = df[df['pIspG'] == '-']

    target_negative_size = int(len(positive_df) * ratio)
    
    n_samples = min(target_negative_size, len(negative_df))
    negative_sampled = negative_df.sample(n=n_samples, random_state=random_state)

    balanced_df = pd.concat([positive_df, negative_sampled])
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    used_ids = set(balanced_df['GenBankID'])
    remaining_df = df[~df['GenBankID'].isin(used_ids)].reset_index(drop=True)

    print("Original distribution:")
    print(df['pIspG'].value_counts())
    print("\nBalanced distribution:")
    print(balanced_df['pIspG'].value_counts())
    print("\nRemaining distribution:")
    print(remaining_df['pIspG'].value_counts())

    return balanced_df, remaining_df


def cif_to_pdb(cif_path: str,
               output_dir: str = "converted_pdbs",
               model_id: str = "model") -> str:
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build the output filename: keep the cif basename, swap extension
    base = os.path.splitext(os.path.basename(cif_path))[0]
    pdb_filename = base + ".pdb"
    pdb_path = os.path.join(output_dir, pdb_filename)

    # Parse the CIF and write out PDB
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(model_id, cif_path)

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)

    return pdb_path

new_pdb = cif_to_pdb(cif_path="raw_data/fold_wp_012464304_1_with_flda_model_0.cif", output_dir="pdb_converted")
print("Wrote PDB to:", new_pdb)