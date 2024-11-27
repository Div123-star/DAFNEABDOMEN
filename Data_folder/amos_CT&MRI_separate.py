import os
import nibabel as nib
import numpy as np
import shutil

# Function to determine if the image is CT or MRI
def is_ct_or_mri(image_path):
    try:
        img = nib.load(image_path)
        data = img.get_fdata()
        min_val, max_val = np.min(data), np.max(data)

        # Assuming CT has a broader range of values, typically from -1000 to 3000 Hounsfield units
        if min_val < -500 and max_val > 1000:
            return "CT"
        else:
            return "MRI"
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return "Unknown"

# Function to analyze images and write MRI and CT filenames to separate text files
def analyze_and_save_images(directory_path, ct_file_name, mri_file_name):
    with open(ct_file_name, "w") as ct_file, open(mri_file_name, "w") as mri_file:
        for filename in os.listdir(directory_path):
            if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                image_path = os.path.join(directory_path, filename)
                modality = is_ct_or_mri(image_path)

                if modality == "CT":
                    ct_file.write(f"{filename}\n")
                elif modality == "MRI":
                    mri_file.write(f"{filename}\n")

                print(f"{filename}: {modality}")

# Function to move images and corresponding label files to separate folders
def move_images_and_labels(image_source_folder, label_source_folder, file_list, image_destination_folder, label_destination_folder):
    os.makedirs(image_destination_folder, exist_ok=True)
    os.makedirs(label_destination_folder, exist_ok=True)

    with open(file_list, "r") as file:
        for line in file:
            image_name = line.strip()  # Get the image filename without whitespace
            image_path = os.path.join(image_source_folder, image_name)  # Construct the full image path

            # Construct the corresponding label filename and path
            label_name = image_name
            label_path = os.path.join(label_source_folder, label_name)

            # Move the image file
            if os.path.exists(image_path):
                shutil.move(image_path, image_destination_folder)
                print(f"Moved {image_path} to {image_destination_folder}")
            else:
                print(f"Image file not found: {image_path}")

            # Move the corresponding label file
            if os.path.exists(label_path):
                shutil.move(label_path, label_destination_folder)
                print(f"Moved {label_path} to {label_destination_folder}")
            else:
                print(f"Label file not found: {label_path}")

# Analyze and separate images and labels for training, testing, and validation datasets
def main():
    base_path = "/path/to/your/amos22/dataset"  # Replace with the correct path to your AMOS22 dataset

    # Analyze training images
    analyze_and_save_images(
        os.path.join(base_path, "imagesTr"),
        "ct_files_tr.txt",
        "mri_files_tr.txt"
    )
    move_images_and_labels(
        os.path.join(base_path, "imagesTr"),
        os.path.join(base_path, "labelsTr"),
        "mri_files_tr.txt",
        os.path.join(base_path, "imagesTr_mri"),
        os.path.join(base_path, "labelsTr_mri")
    )
    move_images_and_labels(
        os.path.join(base_path, "imagesTr"),
        os.path.join(base_path, "labelsTr"),
        "ct_files_tr.txt",
        os.path.join(base_path, "imagesTr_ct"),
        os.path.join(base_path, "labelsTr_ct")
    )

    # Analyze testing images
    analyze_and_save_images(
        os.path.join(base_path, "imagesTs"),
        "ct_files_ts.txt",
        "mri_files_ts.txt"
    )
    move_images_and_labels(
        os.path.join(base_path, "imagesTs"),
        os.path.join(base_path, "labelsTs"),
        "mri_files_ts.txt",
        os.path.join(base_path, "imagesTs_mri"),
        os.path.join(base_path, "labelsTs_mri")
    )
    move_images_and_labels(
        os.path.join(base_path, "imagesTs"),
        os.path.join(base_path, "labelsTs"),
        "ct_files_ts.txt",
        os.path.join(base_path, "imagesTs_ct"),
        os.path.join(base_path, "labelsTs_ct")
    )

    # Analyze validation images
    analyze_and_save_images(
        os.path.join(base_path, "imagesVa"),
        "ct_files_va.txt",
        "mri_files_va.txt"
    )
    move_images_and_labels(
        os.path.join(base_path, "imagesVa"),
        os.path.join(base_path, "labelsVa"),
        "mri_files_va.txt",
        os.path.join(base_path, "imagesVa_mri"),
        os.path.join(base_path, "labelsVa_mri")
    )
    move_images_and_labels(
        os.path.join(base_path, "imagesVa"),
        os.path.join(base_path, "labelsVa"),
        "ct_files_va.txt",
        os.path.join(base_path, "imagesVa_ct"),
        os.path.join(base_path, "labelsVa_ct")
    )

if __name__ == "__main__":
    main()
