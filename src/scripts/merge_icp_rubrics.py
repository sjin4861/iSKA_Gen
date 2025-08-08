
import json
import os
from collections import defaultdict
import datetime

# Define the base path and output directory
base_path = "src/data/pairwise_data/train/v3/icp"
output_dir = "src/data/rm_training/icp/2025-08-02"
os.makedirs(output_dir, exist_ok=True)

# Get the list of model pairs
model_pairs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# Dictionary to hold data for each rubric
rubric_data = defaultdict(list)

# Loop through each model pair and rubric
for pair in model_pairs:
    pair_path = os.path.join(base_path, pair)
    if not os.path.isdir(pair_path):
        continue
    rubrics = [r for r in os.listdir(pair_path) if os.path.isdir(os.path.join(pair_path, r))]
    for rubric in rubrics:
        json_file_path = os.path.join(pair_path, rubric, "rm_pairwise.json")
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                rubric_data[rubric].extend(data)

# Write the merged data to new files
today_str = datetime.date.today().strftime("%Y%m%d")
for rubric, data in rubric_data.items():
    output_filename = os.path.join(output_dir, f"ICP_{rubric}_{today_str}.json")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Successfully merged {len(data)} pairs for rubric '{rubric}' into {output_filename}")
