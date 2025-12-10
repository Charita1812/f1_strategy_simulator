import zipfile
import os
import pandas as pd
import pickle
import io


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
    Uses pickle.loads() with read bytes to avoid file object issues.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(filename) as f:
                # Read all bytes first, then unpickle
                # This avoids issues with file-like object compatibility
                data = f.read()
                return pickle.loads(data)
    except Exception as e:
        print(f"Error loading {filename} from {zip_path}: {e}")
        # Fallback: extract and load from disk
        print(f"Attempting fallback method...")
        extract_dir = os.path.dirname(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extract(filename, extract_dir)
        
        extracted_path = os.path.join(extract_dir, filename)
        with open(extracted_path, 'rb') as f:
            model = pickle.load(f)
        
        # Clean up extracted file
        try:
            os.remove(extracted_path)
        except:
            pass
        
        return model


def load_csv(zip_path: str, filename: str):
    """
    Load CSV directly from a ZIP without extracting.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(filename) as f:
            return pd.read_csv(f)
