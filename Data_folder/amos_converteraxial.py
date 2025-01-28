import os
import numpy as np
import nibabel as nib
import zipfile

# Constants for organ label values in the AMOS dataset
ORGAN_LABELS = {
    1: 'spleen',
    2: 'right_kidney',
    3: 'left_kidney',
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

def read_image(path):
    """Read image using nibabel and return numpy array and resolution."""
    image = nib.load(path)
    image_array = np.fliplr(np.flipud(np.transpose(image.get_fdata(), (1,0,2) )))
    resolution = image.header.get_zooms()
    return image_array, resolution

def convert_and_save_to_npz(image_folder, label_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for image_filename in os.listdir(image_folder):
        if image_filename.endswith('.nii') or image_filename.endswith('.nii.gz'):
            image_path = os.path.join(image_folder, image_filename)
            label_filename = image_filename.replace("images", "labels")  # Adjust filename to match labels
            label_path = os.path.join(label_folder, label_filename)

            if os.path.exists(label_path):
                # Load image and label data
                img_array, resolution = read_image(image_path)
                label_array, _ = read_image(label_path)

                # Prepare data dictionary
                data = {'data': img_array.astype(np.float32), 'resolution': resolution}

                # Generate organ masks
                for label_value, organ_name in ORGAN_LABELS.items():
                    organ_mask = (label_array == label_value).astype(np.uint8)
                    if np.any(organ_mask):
                        data[f'mask_{organ_name}'] = organ_mask

                # Save data to .npz file
                npz_filename = image_filename.replace('.nii.gz', '.npz').replace('.nii', '.npz')
                npz_path = os.path.join(output_folder, npz_filename)
                np.savez_compressed(npz_path, **data)
                print(f"Saved {npz_path}")
            else:
                print(f"Label file not found for {image_filename}")

def zip_npz_files(source_folder, output_zip_file):
    with zipfile.ZipFile(output_zip_file, 'w') as zipf:
        for root, _, files in os.walk(source_folder):
            for file in files:
                if file.endswith('.npz'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=os.path.relpath(file_path, source_folder))
                    print(f"Added {file_path} to {output_zip_file}")

def main():
    base_path = "/Users/dibya/dafne/MyThesisDatasets/amos22/MRI_data"  # Path where MRI_data folder will be created
    output_folder = os.path.join(base_path, "npz_files")
    output_zip_file = os.path.join(base_path, "MRI_data.zip")

    #convert_and_save_to_npz("/Users/dibya/dafne/MyThesisDatasets/amos22/test_data", "/Users/dibya/dafne/MyThesisDatasets/amos22/test_labels", '/Users/dibya/dafne/MyThesisDatasets/test_output/')
    # Convert and save images and labels from each MRI folder
    convert_and_save_to_npz("/Users/dibya/dafne/MyThesisDatasets/amos22/imagesTr_mri", "/Users/dibya/dafne/MyThesisDatasets/amos22/labelsTr_mri", output_folder)
    convert_and_save_to_npz("/Users/dibya/dafne/MyThesisDatasets/amos22/imagesTs_mri", "/Users/dibya/dafne/MyThesisDatasets/amos22/labelsTs_mri", output_folder)
    convert_and_save_to_npz("/Users/dibya/dafne/MyThesisDatasets/amos22/imagesVa_mri", "/Users/dibya/dafne/MyThesisDatasets/amos22/labelsVa_mri", output_folder)


if __name__ == "__main__":
    main()