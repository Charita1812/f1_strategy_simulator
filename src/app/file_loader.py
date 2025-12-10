import zipfile
import os
import pandas as pd
import pickle


def extract_if_needed(zip_path: str, extract_to: str):
    """
    Extracts the zip file only if the target file does not exist.
    """
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path} → {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(os.path.dirname(zip_path))
    return extract_to



def load_pickle(zip_path: str, filename: str):
    """
    Load pickle model directly from a ZIP without extracting.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(filename) as f:
            return pickle.load(BytesIO(f.read()))

def load_csv(zip_path: str, filename: str):
    """
    Load CSV directly from a ZIP without extracting.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(filename) as f:
            return pd.read_csv(f)

