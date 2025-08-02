from typing import List
import numpy as np
import re
import os
from PIL import Image
import matplotlib.pyplot as plt

from pycaf import Experiment


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def _get_image_from_file(
    filepath: str
) -> np.ndarray:
    # BUG FIX: Python was not realeasing the file resource for deletion
    with Image.open(filepath, 'r') as imagefile:
        image = np.array(imagefile, dtype=float)
    return image


def read_images(
    image_dirpath: str
) -> np.ndarray:
    images = []
    filenames = list(os.listdir(image_dirpath))
    filenames.sort(key=natural_keys)
    for filename in filenames:
        if '.tif' in filename:
            imagepath = os.path.join(image_dirpath, filename)
            image = _get_image_from_file(imagepath)
            images.append(image)
    return np.array(images)


def delete_images(
    image_dirpath: str
) -> None:
    for filename in os.listdir(image_dirpath):
        if '.tif' in filename:
            os.remove(os.path.join(image_dirpath, filename))
    return None


def iterative_run(
    experiment: Experiment,
    scripts: List[str],
    parameters: List[str],
    values: List[str]
) -> List[np.ndarray]:
    image_dirpath: str = experiment.config["temp_image_path"]
    images: List[np.ndarray] = []
    n_iter: int = len(scripts)
    for k in range(n_iter):
        experiment.scan(
            scripts[k],
            motmaster_parameters_with_values={
                parameters[k]: [values[k]]
            },
            n_iter=1
        )
        try:
            _images = read_images(image_dirpath)
        except Exception as e:
            print(f"Exception {e} occured in during reading images")
        try:
            delete_images(image_dirpath)
        except Exception as e:
            print(f"Exception {e} occured in during deleting images")
        if len(_images) >= 2:
            image = np.sum(_images[0::2, :, :] - _images[1::2, :, :], axis=0)
            images.append(image)
    return images
