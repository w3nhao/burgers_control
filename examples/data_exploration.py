import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_squence_data, train_file_path

data = get_squence_data(train_file_path)

print(data)