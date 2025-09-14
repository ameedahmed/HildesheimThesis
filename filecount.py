import os
total_count_files = 0
for root_dir, cur_dir, files in os.walk(r'/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/cropped_dataset/train'):
    total_count_files += len(files)

directories = ['/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/cropped_dataset/train/cropped_train_academicism_dataset',
                '/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/cropped_dataset/train/cropped_train_baroque_dataset',
                '/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/cropped_dataset/train/cropped_train_neoclassicism_dataset',
                '/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/cropped_dataset/train/cropped_train_orientalism_dataset',
                '/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/cropped_dataset/train/cropped_train_realism_dataset',
                '/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/cropped_dataset/train/cropped_train_Ukiyo_e_dataset']
list_files_in_dir = {}
for directory in directories:
    count = 0
    for file in os.scandir(directory):
        if file.is_file():
            count += 1
    list_files_in_dir[directory] = count

#Print the weights of each directory by dividing the number of files within each directory with the total files
weights = []
for directory in directories:
    weights.append(total_count_files/list_files_in_dir[directory])

print(weights)