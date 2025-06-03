import pandas as pd
import os, csv
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, MMCIFIO
import glob

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
        writer.writerow(["ID", "measurement_value"])
        for file in sorted(path.iterdir()):
            if file.is_file():
                writer.writerow([file.stem, 0])

def merge_chains(input_file: str, output_file: str):
    parser = MMCIFParser(QUIET=True) if input_file.endswith(".cif") else PDBParser(QUIET=True)
    structure = parser.get_structure("model", input_file)

    model = structure[0]
    if "A" not in model or "B" not in model:
        print(f"[skip] {input_file} – chains A and B not both present")

    chainA, chainB = model["A"], model["B"]

    # residue offset = last residue number in chain A
    offset = chainA.get_list()[-1].id[1]

    # move residues from B to A with new numbers
    for res in chainB.get_residues():
        het, seq, icode = res.id
        res.id = (het, seq + offset, icode)
        chainA.add(res)
    model.detach_child("B")

    # write out as mmCIF
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_file)
    print(f"[ok]  {input_file}  →  {output_file}")

# for filepath in glob.glob("dimers_test/*"):
#     merge_chains(filepath, filepath.replace('.cif', '_merged.cif'))

# folder = Path("dimers_test")
# for f in folder.iterdir():
#     if f.is_file() and not f.name.endswith("_merged.cif"):
#         f.unlink()  

# pdbs = convert_all_cif_to_pdb("dimers_test", output_dir="pdb_converted")
# make_affinity_csv("pdb_converted", out_csv="binding_affinities_dimers.csv")

basic_features = pd.read_csv('datasets/final_feature_matrix_v2.csv')
msa_features = pd.read_csv('ispG_MSA_features_full.csv')
deeprank_features = pd.read_csv('deeprank_features.csv')
phylo_tree = pd.read_csv('ispG_distances_to_AAC75568.1.csv')
print(phylo_tree)

new = basic_features.rename(columns={'GenBankID': 'id'})
# 2) First merge: left ↔ middle
#merged = msa_features.merge(deeprank_features, on='id', how= 'inner', suffixes=("_msa", ""))
merged = msa_features.merge(new, on='id', how='inner', suffixes=("", "_gb"))
merged = merged.merge(phylo_tree, on = 'id', how = 'inner')

merged.to_csv("all_merged_wo_deeprank.csv", index = False)
print(merged.head())