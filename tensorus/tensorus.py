import os
import json
import shutil

# Define the storage directory relative to the repository root
STORAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'storage')
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

def upload_tensor(file_path):
    """
    Uploads a file by copying it into the storage directory.
    :param file_path: The path of the file to upload.
    :return: The destination path where the file is stored.
    """
    file_name = os.path.basename(file_path)
    dest_path = os.path.join(STORAGE_DIR, file_name)
    shutil.copy(file_path, dest_path)
    return dest_path

def load_tensor(file_name):
    """
    Loads tensor data from a file stored in the storage directory.
    Assumes the file contains JSON-encoded tensor data.
    :param file_name: The name of the file in the storage directory.
    :return: The parsed tensor data.
    """
    file_path = os.path.join(STORAGE_DIR, file_name)
    with open(file_path, 'r') as f:
        return json.load(f)
