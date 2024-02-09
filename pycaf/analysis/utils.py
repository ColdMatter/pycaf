from typing import Any, Dict, List, Tuple
from zipfile import ZipFile
from PIL import Image
from pathlib import Path
import re
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from .models import (
    Pattern,
    GaussianFitWithOffset,
    GaussianFitWithOffset2D
)
from .curve_fitting import (
    fit_gaussian_with_offset,
    fit_gaussian_with_offset_2D
)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_zip_archive(
    root: str,
    year: int,
    month: int,
    day: int,
    file_no: int,
    prefix: str,
    mode: str = "r"
) -> ZipFile:
    month_dict: Dict[int, str] = \
        {
            1: "01Jan",
            2: "02Feb",
            3: "03Mar",
            4: "04Apr",
            5: "05May",
            6: "06Jun",
            7: "07Jul",
            8: "08Aug",
            9: "09Sep",
            10: "10Oct",
            11: "11Nov",
            12: "12Dec"
        }
    day_filled: str = str(day).zfill(2)
    year_filled: str = str(year)[-2:]
    file_filled: str = str(file_no).zfill(3)
    month_filled: str = month_dict[month][-3:]
    rootpath = Path(root)
    filepath = rootpath.joinpath(
        str(year),
        month_dict[month],
        day_filled,
        f"{prefix}{day_filled}{month_filled}{year_filled}00_{file_filled}.zip"
    )
    return ZipFile(filepath, mode=mode)


def create_file_list(
    file_no_start: int,
    file_no_stop: int = None,
    background_file_start: int = None,
    background_file_stop: int = None,
    exclude_files: List[int] = []
) -> Tuple[List[int], List[int]]:
    files, background_files = [], []
    if file_no_start and not file_no_stop:
        files.append(file_no_start)
        background_files.append(background_file_start)
    if file_no_start and file_no_stop:
        for i in range(file_no_start, file_no_stop+1):
            if i not in exclude_files:
                files.append(i)
        if background_file_start and not background_file_stop:
            for i in range(file_no_start, file_no_stop+1):
                if i not in exclude_files:
                    background_files.append(background_file_start)
        elif background_file_start and background_file_stop:
            for i in range(background_file_start, background_file_stop+1):
                if i not in exclude_files:
                    background_files.append(i)
    return files, background_files


def remote_image_injector(
    archives: List[ZipFile],
    n_images: int,
    remote_path: str
) -> None:
    imgs = os.listdir(remote_path)
    imgs.sort(key=natural_keys)
    if len(imgs) == len(archives)*n_images:
        print("Inserting images to the zip files...")
        i = 0
        for archive in archives:
            files = archive.namelist()
            for _ in range(n_images):
                if imgs[i] not in files:
                    archive.write(os.path.join(remote_path, imgs[i]), imgs[i])
                    i += 1
        for img in imgs:
            os.remove(os.path.join(remote_path, img))
        for archive in archives:
            archive.close()
    elif len(imgs) == 0:
        print("No Image to insert")
    elif len(imgs) < len(archives)*n_images:
        print("There seems to be less number of images than required!")
    elif len(imgs) > len(archives)*n_images:
        print("There are more images than expected!")
    return None


def extract_pattern_from_json_string(
    line: str
) -> Tuple[Dict[str, int], np.ndarray]:
    data = json.loads(line)
    patterns = []
    for value in data.values():
        channels = value["channels"]
        split_patterns = value["pattern"].split("\n")
        for split_pattern in split_patterns:
            if len(split_pattern):
                _pattern_tt = split_pattern.split("\t\t")
                if len(_pattern_tt) == 2:
                    pattern = [
                        _pattern_tt[0],
                        *_pattern_tt[1].split("\t")
                    ]
                    patterns.append(pattern[:-1])
                else:
                    pattern = _pattern_tt[0].split("\t")
                    patterns.append(pattern[:-1])
    patterns = np.array(patterns[1:], dtype=str)
    return channels, patterns


def read_digital_patterns_from_zip(
    archive: ZipFile,
    close: bool = True
) -> Dict[str, Pattern]:
    parameters = read_parameters_from_zip(archive)
    full_time = parameters["PatternLength"]
    channels: Dict[str, Pattern] = {}
    for filename in archive.namelist():
        if filename[-19:] == "digitalPattern.json":
            with archive.open(filename) as f:
                lines = f.readlines()
                for line in lines:
                    _channels, patterns = \
                        extract_pattern_from_json_string(line)
                    timings = np.array(patterns[:, 0], dtype=int)
                    for name, index in _channels.items():
                        filled_timings = np.empty(0, dtype=int)
                        filled_patterns = np.empty(0, dtype=int)
                        next_seq = 0
                        for i, timing in enumerate(timings):
                            seq = patterns[i, index+1]
                            if seq == "U":
                                filled_timings = \
                                    np.append(filled_timings, timing-1)
                                filled_patterns = np.append(filled_patterns, 0)
                                filled_timings = \
                                    np.append(filled_timings, timing)
                                filled_patterns = np.append(filled_patterns, 5)
                                next_seq = 5
                            elif seq == "D":
                                filled_timings = \
                                    np.append(filled_timings, timing-1)
                                filled_patterns = np.append(filled_patterns, 5)
                                filled_timings = \
                                    np.append(filled_timings, timing)
                                filled_patterns = np.append(filled_patterns, 0)
                                next_seq = 0
                            elif seq == "-":
                                filled_timings = \
                                    np.append(filled_timings, timing)
                                filled_patterns = \
                                    np.append(filled_patterns, next_seq)
                        filled_timings = np.append(filled_timings, full_time)
                        filled_patterns = np.append(filled_patterns, next_seq)
                        channels[name] = Pattern(
                            name=name,
                            time=filled_timings,
                            event=filled_patterns
                        )
    if close:
        archive.close()
    return channels


def read_analog_patterns_from_zip(
    archive: ZipFile,
    close: bool = True
) -> Dict[str, Pattern]:
    parameters = read_parameters_from_zip(archive, False)
    full_time = parameters["PatternLength"]
    offset = int(parameters["TCLBlockStart"])
    reader = {}
    for filename in archive.namelist():
        if filename[-18:] == "analogPattern.json":
            with archive.open(filename) as f:
                reader.update(json.load(f))
    channels: Dict[str, Pattern] = {}
    for name, channel_data in reader.items():
        timings = np.empty(0, dtype=int)
        voltages = np.empty(0, dtype=float)
        for timing, voltage in channel_data.items():
            timings = np.append(timings, int(timing)+offset)
            voltages = np.append(voltages, float(voltage))
        filled_timings = np.empty(0, dtype=int)
        filled_voltages = np.empty(0, dtype=float)
        if len(timings) == 1:
            filled_timings = np.append(filled_timings, timings[0])
            filled_voltages = np.append(filled_voltages, voltages[0])
            filled_timings = np.append(filled_timings, full_time)
            filled_voltages = np.append(filled_voltages, voltages[0])
        elif len(timings) > 1:
            for i in range(len(timings)-1):
                dt = timings[i+1]-timings[i]
                if dt > 1:
                    filled_timings = np.append(filled_timings, timings[i])
                    filled_voltages = np.append(filled_voltages, voltages[i])
                    filled_timings = np.append(filled_timings, timings[i+1]-1)
                    filled_voltages = np.append(filled_voltages, voltages[i])
                    filled_timings = np.append(filled_timings, timings[i+1])
                    filled_voltages = np.append(filled_voltages, voltages[i+1])
                else:
                    filled_timings = np.append(filled_timings, timings[i])
                    filled_voltages = np.append(filled_voltages, voltages[i])
            filled_timings = np.append(filled_timings, full_time)
            filled_voltages = np.append(filled_voltages, voltages[-1])
        channels[name] = Pattern(
            name=name,
            time=filled_timings,
            event=filled_voltages
        )
    if close:
        archive.close()
    return channels


def read_images_from_zip(
    archive: ZipFile,
    close: bool = True
) -> np.ndarray:
    images = []
    filenames = archive.namelist()
    filenames.sort(key=natural_keys)
    for filename in filenames:
        if filename[-3:] == "tif":
            with archive.open(filename) as image_file:
                images.append(
                    np.array(
                        Image.open(image_file),
                        dtype=float
                    )
                )
    if close:
        archive.close()
    return np.array(images)


def read_parameters_from_zip(
    archive: ZipFile,
    close: bool = True
) -> Dict[str, Any]:
    parameters = {}
    for filename in archive.namelist():
        if filename[-14:] == "parameters.txt":
            with archive.open(filename) as parameter_file:
                script_parameters = parameter_file.readlines()
                for line in script_parameters:
                    name, value, _ = line.split(b"\t")
                    parameters[name.decode("utf-8")] = float(value)
        elif filename[-18:] == "hardwareReport.txt":
            with archive.open(filename) as hardware_file:
                hardware_parameters = hardware_file.readlines()
                for line in hardware_parameters:
                    name, value, _ = line.split(b"\t")
                    if value.isdigit():
                        parameters[name.decode("utf-8")] = float(value)
    if close:
        archive.close()
    return parameters


def read_time_of_flight_from_zip(
    archive: ZipFile,
    close: bool = True
) -> Tuple[int, np.ndarray]:
    tofs = []
    sampling_rate: int = 0
    sorted_filenames = archive.namelist()
    sorted_filenames.sort(key=natural_keys)
    for filename in sorted_filenames:
        if filename[0:3] == "Tof":
            with archive.open(filename) as tof_file:
                lines: List[bytes] = tof_file.readlines()
                tofs.append(lines[1:])
                sampling_rate: int = int(
                    lines[0].decode("utf-8").split(",")[0].split(":")[-1]
                )
    if len(tofs) > 1:
        tofs = np.array(tofs, dtype=float).mean(axis=0)
    if close:
        archive.close()
    return sampling_rate, tofs


def read_time_of_flight_from_zip_no_mean(
    archive: ZipFile,
    close: bool = True
) -> Tuple[int, np.ndarray]:
    tofs = []
    sampling_rate: int = 0
    sorted_filenames = archive.namelist()
    sorted_filenames.sort(key=natural_keys)
    for filename in sorted_filenames:
        if filename[0:3] == "Tof":
            with archive.open(filename) as tof_file:
                lines: List[bytes] = tof_file.readlines()
                tofs.append(lines[1:])
                sampling_rate: int = int(
                    lines[0].decode("utf-8").split(",")[0].split(":")[-1]
                )
    if len(tofs) > 1:
        tofs = np.array(tofs, dtype=float)
    if close:
        archive.close()
    return sampling_rate, tofs


def smooth_time_of_flight(
    tofs: np.ndarray,
    points: int = 51,
    polynomial: int = 2
) -> np.ndarray:
    return savgol_filter(tofs, points, polynomial)


def remove_outliers(
    tof: np.ndarray,
    threshold: float = 5.0
) -> np.ndarray:
    '''
    Remove outliers in tof and replace it by average of adjacent points.
    Input 1d tof array, output 1d modified array.
    Use lower threshold to apply heavier filtering
    '''
    modified_tof = tof
    length = len(tof)

    tof_anomaly = []
    for i in range(length - 2):
        tof_anomaly.append(abs(tof[i] - 2 * tof[i+1] + tof[i+2]))

    # Use Modified Z-Score method to identify outliers of tof_anomaly
    # Calculate the median and the median absolute deviation (MAD)
    median = np.median(tof_anomaly)
    mad = np.median(np.abs(tof_anomaly - median))

    # Calculate the Modified Z-scores for each data point
    modified_z_scores = 0.6745 * (tof_anomaly - median) / mad

    # Find the indices of the data points with Modified
    # Z-score above the threshold (outliers)
    outliers_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
    normal_indeces = np.arange(0, length)
    normal_indeces = [x for x in normal_indeces if x not in outliers_indices]

    # Modify tof by replaceing outliers with average of adjacent points
    for index in outliers_indices:
        index_1 = index
        index_2 = index + 2

        while index_1 in outliers_indices:
            index_1 -= 1

        while index_2 in outliers_indices:
            index_2 += 1

        if index_1 < 0:
            index_1 = 0

        if index_2 >= length:
            index_2 = length - 1

        modified_tof[index + 1] = (tof[index_1] + tof[index_2]) / 2
    return modified_tof


def crop_image(
    image: np.ndarray,
    centre: Tuple[int, int],
    height: int,
    width: int
) -> np.ndarray:
    hstart = int(centre[0]-height/2)
    hstop = int(centre[0]+height/2)
    vstart = int(centre[1]-width/2)
    vstop = int(centre[1]+width/2)
    return image[hstart:hstop, vstart:vstop]


def crop_images(
    images: np.ndarray,
    centre: Tuple[int, int],
    height: int,
    width: int
) -> np.ndarray:
    hstart = int(centre[0]-height/2)
    hstop = int(centre[0]+height/2)
    vstart = int(centre[1]-width/2)
    vstop = int(centre[1]+width/2)
    return images[:, hstart:hstop, vstart:vstop]


def bin_image(
    image: np.ndarray,
    h_bin: int = 2,
    v_bin: int = 2
) -> np.ndarray:
    _img = 0
    for i in range(v_bin):
        for j in range(h_bin):
            _img += image[i::v_bin, j::h_bin]
    return _img


def calculate_molecule_number_from_fluorescent_images(
    images: np.ndarray,
    full_well_capacity: float,
    bits_per_channel: int,
    exposure_time: float,
    gamma: float,
    collection_solid_angle: float,
    eta_q: float
) -> np.ndarray:
    count = np.sum(images, axis=(1, 2))
    photon = (count*full_well_capacity)/((2**bits_per_channel-1)*eta_q)
    number = photon/(exposure_time*gamma*collection_solid_angle)
    return number


def calculate_optical_density_from_absorption_images(
    images: np.ndarray,
    probe_images: np.ndarray,
    background_images: np.ndarray
) -> np.ndarray:
    images -= background_images
    probe_images -= background_images
    images[images <= 0] = 1.0
    od = -np.log(images/probe_images)
    od[np.isnan(od)] = 0.0
    od[od == -np.inf] = 0.0
    od[od == np.inf] = 0.0
    return od


def calculate_atom_number_from_absorption_images(
    optical_density: np.ndarray,
    pixel_size: float,
    bin_size: int,
    magnification: float,
    saturation: float
) -> np.ndarray:
    count = np.sum(optical_density, axis=(1, 2))
    number = count*saturation*(pixel_size*bin_size/magnification)**2
    return number


def calculate_cloud_size_from_image_1d_gaussian(
    image: np.ndarray,
    pixel_size: float,
    bin_size: int,
    magnification: float,
) -> Tuple[GaussianFitWithOffset, GaussianFitWithOffset]:
    vertical_integrate = np.sum(image, axis=0)
    horizontal_integrate = np.sum(image, axis=1)
    scale = (pixel_size*bin_size/magnification)
    vertival_fit = fit_gaussian_with_offset(
        scale*np.arange(0, len(vertical_integrate)),
        vertical_integrate
    )
    horizontal_fit = fit_gaussian_with_offset(
        scale*np.arange(0, len(horizontal_integrate)),
        horizontal_integrate
    )
    return (vertival_fit, horizontal_fit)


def calculate_cloud_size_from_image_2d_gaussian(
    image: np.ndarray,
    pixel_size: float,
    bin_size: int,
    magnification: float,
) -> GaussianFitWithOffset2D:
    scale = bin_size/magnification*pixel_size
    x = np.arange(0, image.shape[1])*scale
    y = np.arange(0, image.shape[0])*scale
    fit = fit_gaussian_with_offset_2D(x, y, image)
    return fit


def calculate_temperature(
    archives: List[ZipFile],
    temporal_parameter: str,
) -> None:
    return None


def calculate_lifetime(
    archives: List[ZipFile],
    temporal_parameter: str,
) -> None:
    return None
