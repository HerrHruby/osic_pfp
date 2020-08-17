import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt


ROOT_TRAIN_DIR = r"/media/matthew/Datasets/Kaggle/OSIC_PFP/osic-pulmonary-fibrosis-progression/train"
ROOT_TEST_DIR = r"/media/matthew/Datasets/Kaggle/OSIC_PFP/osic-pulmonary-fibrosis-progression/test"


def clip_border(image: np.ndarray) -> np.ndarray:
    bounds = np.array(np.nonzero(~(image == 0)))
    tl = np.min(bounds, axis=1)
    br = np.max(bounds, axis=1)

    return image[tl[0]:br[0], tl[1]:br[1]]


def rescale_for_lungs(image: np.ndarray, meta_image: pydicom.FileDataset) -> np.ndarray:
    hounsfield_units = image * meta_image.RescaleSlope + meta_image.RescaleIntercept
    lung_min = -600
    lung_max = -400
    return ((hounsfield_units - lung_min) / (lung_max - lung_min)) * 255


def preprocess(medical_image: pydicom.FileDataset) -> np.ndarray:
    return rescale_for_lungs(clip_border(medical_image.pixel_array), medical_image)


def show_dir_images(path):
    fig, ax = plt.subplots(nrows=5, ncols=5)
    plt.style.use("grayscale")
    for i, patient in enumerate(os.listdir(path)):
        patient_path = os.path.join(path, patient)
        sample_file_path = os.path.join(patient_path, os.listdir(patient_path)[0])
        sample_file = pydicom.dcmread(sample_file_path)
        try:
            ax[i // 5][i % 5].imshow(preprocess(sample_file), cmap=plt.cm.jet)
        except RuntimeError:
            print(f"Failed to load image for patient {patient}")
        if i >= 24:
            break
    plt.show()


def main():
    show_dir_images(ROOT_TRAIN_DIR)


if __name__ == '__main__':
    main()
