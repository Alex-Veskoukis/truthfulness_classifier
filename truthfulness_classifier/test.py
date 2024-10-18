test_path = "E:/results/test_file.txt"
with open(test_path, 'w') as f:
    f.write("This is a test.")

import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Device Count:", torch.cuda.device_count())
