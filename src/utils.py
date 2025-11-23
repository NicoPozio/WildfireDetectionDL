import torch
import numpy as np
import random
import os
import zipfile

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def extract_zip(source_path, dest_folder):
    """
    Extracts a zip file to a destination with a progress bar.
    """
    try:
        with zipfile.ZipFile(source_path, 'r') as zip_ref:
            # Get list of files for the progress bar
            members = zip_ref.infolist()
            print(f"Extracting {os.path.basename(source_path)} to {dest_folder}...")
            
            for member in tqdm(members, desc="Unzipping", unit="files"):
                zip_ref.extract(member, dest_folder)
                
        print(f" -> Success: Extracted to {dest_folder}")
        return True
    except zipfile.BadZipFile:
        print(f" -> ERROR: {source_path} is corrupted.")
        return False
