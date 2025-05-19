import h5py
import pandas as pd
from pathlib import Path

def hdf5_to_df(h5_path: Path, dataset_key: str) -> pd.DataFrame:
    """Load a single HDF5 dataset into a pandas DataFrame."""
    with h5py.File(h5_path, "r") as h5:
        data = h5[dataset_key][()]            # numpy array
        if data.ndim == 1:
            df = pd.DataFrame(data, columns=[dataset_key])
        else:
            cols = [f"{dataset_key}_{i}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=cols)
        # Optional: add a column with file name
        df.insert(0, "source_file", h5_path.name)
        return df

def merge_hdf5_folder(folder: str,
                      dataset_key: str,
                      out_file: str = "combined_features.parquet") -> pd.DataFrame:
    """Read all .h5/.hdf5 in folder, merge their <dataset_key> datasets, save one file."""
    folder = Path(folder)
    dfs = []
    for f in sorted(folder.glob("*.h*5")):     # matches .h5 and .hdf5
        dfs.append(hdf5_to_df(f, dataset_key))
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(out_file)                # or .to_csv(...)
    return merged

# ------------------------------------------------------------------
# Example usage:
#   each HDF5 file has a dataset called 'node_feat' (n_nodes × n_features)
#   and you want to stack all rows end-to-end:

all_features = merge_hdf5_folder("interaction_main_A", dataset_key="id")
print(all_features.shape)
