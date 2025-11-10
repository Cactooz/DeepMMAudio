import pandas as pd
from pathlib import Path
import os

def remove_unlisted_videos(tsv_file_path, folder_path):
    df = pd.read_csv(tsv_file_path, sep='\t')
    valid_ids = set(df['id'].astype(str))

    folder = Path(folder_path)
    extensions = {'.flac'}
    removed_count = 0
    
    # Go through all files in the folder
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            # Get filename without extension as the ID
            file_id = file_path.stem
            
            # Remove file if ID is not in the TSV
            if file_id not in valid_ids:
                print(f"Moving: {file_path.name}")
                # Move file to /removed
                if os.path.exists(folder / "removed") == False:
                    os.makedirs(folder / "removed")
                file_path.rename(folder / "removed" / file_path.name)
                removed_count += 1
    
    print(f"\nTotal files moved: {removed_count}")

if __name__ == "__main__":
    tsv_file = "./MMAudio/sets/audioset_sl.tsv"
    folder = "./data/audio/audioset_sl/"

    remove_unlisted_videos(tsv_file, folder)
