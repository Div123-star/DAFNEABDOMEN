import os
import numpy as np

# Constants for organ label values in the AMOS dataset
ORGAN_LABELS = {
    1: 'spleen',
    2: 'RK',  # Right Kidney
    3: 'LK',  # Left Kidney
    4: 'gallbladder',
    5: 'esophagus',
    6: 'liver',
    7: 'stomach',
    8: 'aorta',
    9: 'inferior_vena_cava',
    10: 'portal_splenic_vein',
    11: 'pancreas',
    12: 'right_adrenal_gland',
    13: 'left_adrenal_gland',
    14: 'duodenum',
    15: 'bladder'
}

def extract_images_and_labels(npz_folder, image_output_folder, label_output_folder):
    """
    Extract images and labels from .npz files and save them into separate folders.
    """
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(label_output_folder, exist_ok=True)

    for npz_file in os.listdir(npz_folder):
        if npz_file.endswith('.npz'):
            file_path = os.path.join(npz_folder, npz_file)
            data = np.load(file_path)

            # Extract and save the image
            image = data['data']
            image_filename = os.path.join(image_output_folder, npz_file.replace('.npz', '_image.npy'))
            np.save(image_filename, image)

            # Extract and save labels
            label = data['label']
            label_filename = os.path.join(label_output_folder, npz_file.replace('.npz', '_label.npy'))
            np.save(label_filename, label)

            print(f"Extracted {npz_file}: Image -> {image_filename}, Label -> {label_filename}")

# Example usage
npz_folder = "/Users/dibya/dafne/MyThesisDatasets/amos22/MRI_data/test_npz"  # Folder with .npz files
image_output_folder = "/Users/dibya/dafne/MyThesisDatasets/amos22/test_images"  # Folder for images
label_output_folder = "/Users/dibya/dafne/MyThesisDatasets/amos22/test_labels"  # Folder for labels

extract_images_and_labels(npz_folder, image_output_folder, label_output_folder)
