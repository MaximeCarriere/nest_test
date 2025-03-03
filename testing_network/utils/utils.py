# utils.py
import os
import scipy.sparse as sp

def ensure_directory_exists(directory, clear=False):
    """
    Ensure that the given directory exists. If it doesn't, create it.
    If `clear=True`, remove all files inside before proceeding.

    Args:
    - directory (str): The directory path to ensure exists.
    - clear (bool): If True, delete all files inside the directory before proceeding.
    """
    if os.path.exists(directory):
        if clear:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove file or symbolic link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove directory and its contents
                except Exception as e:
                    print(f"⚠️ Error deleting {file_path}: {e}")
        else:
            print(f"✅ Directory '{directory}' already exists. No files deleted.")
    else:
        os.makedirs(directory)
        print(f"📁 Created directory '{directory}'.")



def get_matrix_weight(df):
    """ Convert weight data into a sparse matrix. """
    max_val = max(df['source'].max(), df['target'].max())
    min_val = min(df['source'].min(), df['target'].min())

    value_to_take = max_val - min_val
    
    matrix = sp.csr_matrix(
        (df['weight'], (df['source'] - min_val, df['target'] - min_val)),
        shape=(value_to_take + 1, value_to_take + 1)
    ).toarray()
    
    return matrix
