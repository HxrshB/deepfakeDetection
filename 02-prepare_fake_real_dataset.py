import os
from distutils.dir_util import copy_tree
import shutil
import numpy as np
import splitfolders

# Base paths for input and output datasets
prepared_dataset_path = './prepared_dataset/'
print('Creating Directory: ' + prepared_dataset_path)
os.makedirs(prepared_dataset_path, exist_ok=True)

tmp_fake_path = './tmp_fake_faces'
print('Creating Directory: ' + tmp_fake_path)
os.makedirs(tmp_fake_path, exist_ok=True)

real_path = os.path.join(prepared_dataset_path, 'real')
print('Creating Directory: ' + real_path)
os.makedirs(real_path, exist_ok=True)

fake_path = os.path.join(prepared_dataset_path, 'fake')
print('Creating Directory: ' + fake_path)
os.makedirs(fake_path, exist_ok=True)

# Process directories for real and fake faces
def copy_faces(base_input_path, output_real_path, output_fake_path, tmp_fake_faces_path):
    for category in ['real', 'fake']:
        input_path = os.path.join(base_input_path, category)
        for subfolder in os.listdir(input_path):
            subfolder_path = os.path.join(input_path, subfolder, 'faces')
            if os.path.exists(subfolder_path):
                if category == 'real':
                    print('Copying to: ' + output_real_path)
                    copy_tree(subfolder_path, output_real_path)
                elif category == 'fake':
                    print('Copying to: ' + tmp_fake_faces_path)
                    copy_tree(subfolder_path, tmp_fake_faces_path)

# Main logic for face copying
print("Copying faces...")
copy_faces('./output', real_path, fake_path, tmp_fake_path)

# Down-sample fake faces to match real faces or handle imbalance
all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
print('Total Number of Real faces: ', len(all_real_faces))

all_fake_faces = [f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))]
print('Total Number of Fake faces: ', len(all_fake_faces))

if len(all_fake_faces) >= len(all_real_faces):
    random_faces = np.random.choice(all_fake_faces, len(all_real_faces), replace=False)
else:
    print("Fewer fake faces than real faces. Copying all fake faces.")
    random_faces = all_fake_faces

for fname in random_faces:
    src = os.path.join(tmp_fake_path, fname)
    dst = os.path.join(fake_path, fname)
    shutil.copyfile(src, dst)

print('Down-sampling Done!')

# Split into Train/Val/Test folders
splitfolders.ratio(prepared_dataset_path, output='split_dataset', seed=1377, ratio=(.8, .1, .1)) # default values
print('Train/Val/Test Split Done!')
