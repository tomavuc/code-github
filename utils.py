import pandas as pd
import os, csv
from pathlib import Path
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


def convert_all_cif_to_pdb(input_folder: str,
                            output_dir: str = "converted_pdbs",
                            model_id: str = "model") -> list[str]:
    input_path = Path(input_folder)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    parser = MMCIFParser(QUIET=True)
    io = PDBIO()
    
    pdb_paths = []
    for cif_file in input_path.iterdir():
        if cif_file.suffix.lower() == ".cif" and cif_file.is_file():
            base = cif_file.stem
            pdb_file = output_path / f"{base}.pdb"
            structure = parser.get_structure(model_id, str(cif_file))
            io.set_structure(structure)
            io.save(str(pdb_file))
            pdb_paths.append(str(pdb_file))
    return pdb_paths


def make_affinity_csv(folder: str,
                      out_csv: str = "binding_affinities.csv") -> None:

    path = Path(folder)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "binding_affinity"]);
        for file in sorted(path.iterdir()):
            if file.is_file():
                writer.writerow([file.name, 0])

pdbs = convert_all_cif_to_pdb("dimers_with_pec", output_dir="pdb_converted")
make_affinity_csv("pdb_converted", out_csv="binding_affinities.csv")