import os
import numpy as np
import pydicom
import cv2


def read_dicom_images(dicom_folder):
    """Read all DICOM images in a patient's folder and store them in a 3D numpy array."""
    dicom_files = [f for f in sorted(os.listdir(dicom_folder)) if f.endswith('.dcm')]
    images = []

    for file in dicom_files:
        ds = pydicom.dcmread(os.path.join(dicom_folder, file))
        images.append(ds.pixel_array)

    images_3d = np.stack(images, axis=-1)
    pixel_spacing = ds.PixelSpacing
    slice_thickness = ds.SliceThickness
    return images_3d, np.array([pixel_spacing[0], pixel_spacing[1], slice_thickness])


def read_ground_truth_images(ground_truth_folder):
    """Read all ground truth images and store them in a 3D numpy array."""
    ground_truth_files = [f for f in sorted(os.listdir(ground_truth_folder)) if f.endswith('.png')]
    masks = []

    for file in ground_truth_files:
        mask = cv2.imread(os.path.join(ground_truth_folder, file), cv2.IMREAD_GRAYSCALE)
        masks.append(mask)

    masks_3d = np.stack(masks, axis=-1)
    return masks_3d


def split_masks(masks_3d):
    """Split the 3D ground truth masks into separate binary masks for each organ."""
    mask_Liver = ((masks_3d >= 55) & (masks_3d < 70)).astype(np.uint8)
    mask_RK = ((masks_3d >= 110) & (masks_3d < 135)).astype(np.uint8)
    mask_LK = ((masks_3d >= 175) & (masks_3d < 200)).astype(np.uint8)
    mask_Spleen = ((masks_3d >= 240) & (masks_3d <= 255)).astype(np.uint8)

    return mask_Liver, mask_RK, mask_LK, mask_Spleen


def save_patient_data(patient_number, data, mask_Liver, mask_RK, mask_LK, mask_Spleen, resolution, output_folder):
    """Save all arrays into a single compressed npz file in the output folder."""
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    file_name = os.path.join(output_folder, f"{patient_number}.npz")
    np.savez_compressed(file_name, data=data, mask_Liver=mask_Liver, mask_RK=mask_RK,
                        mask_LK=mask_LK, mask_Spleen=mask_Spleen, resolution=resolution)
    print(f"Saved data to {file_name}")


def process_all_patients(num_patients):
    """Process and save data for all patients in a loop."""
    base_path = "/Users/dibya/dafne/MyThesisDatasets/CHAOS_Dataset_for_Dibya/MR"
    output_folder = "/Users/dibya/dafne/MyThesisDatasets/all_patient"

    for patient_number in range(1, num_patients + 1):
        print(f"Processing data for patient {patient_number}...")

        # Construct the full paths using your base path
        dicom_folder = f"{base_path}/{patient_number}/T1DUAL/DICOM_anon/OutPhase"
        ground_truth_folder = f"{base_path}/{patient_number}/T1DUAL/Ground"

        # Check if folders exist to avoid errors
        if os.path.exists(dicom_folder) and os.path.exists(ground_truth_folder):
            # Read DICOM images and get resolution
            data, resolution = read_dicom_images(dicom_folder)

            # Read ground truth images
            masks_3d = read_ground_truth_images(ground_truth_folder)

            # Split masks into binary masks for each organ
            mask_Liver, mask_RK, mask_LK, mask_Spleen = split_masks(masks_3d)

            # Save all data into a npz file in the output folder
            save_patient_data(patient_number, data, mask_Liver, mask_RK, mask_LK, mask_Spleen, resolution,
                              output_folder)
        else:
            print(f"Skipping patient {patient_number}: folders not found")


# Example usage for 36 patients
process_all_patients(36)
