import os
import json

def generate_class_index_json(dataset_dir, output_file):
    class_index = {}
    subfolders = sorted(os.listdir(dataset_dir))
    for idx, folder in enumerate(subfolders):
        print(folder)
        class_name = folder.split('_')[1]  # Extract class name from folder name
        class_index[str(idx)] = [os.path.join(dataset_dir, folder), class_name]
    
    with open(output_file, 'w') as f:
        json.dump(class_index, f, indent=4)

# Example usage
train_dir = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/train_folder_train_val_test_combi"
output_file = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/trainvaltestcombi_artstyle_class_index.json"
generate_class_index_json(train_dir, output_file)