from tensorus import upload_tensor, load_tensor

import sys
if len(sys.argv) < 2:
    print("Usage: python test.py <path-to-tensor-json-file>")
    sys.exit(1)

file_path = sys.argv[1]
dest_path = upload_tensor(file_path)
print("Uploaded file stored at:", dest_path)

# Assuming the uploaded file’s name remains the same
import os
file_name = os.path.basename(file_path)
tensor_data = load_tensor(file_name)
print("Loaded tensor data:", tensor_data)
