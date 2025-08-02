from typing import Any, Dict, List, Tuple
from zipfile import ZipFile
from PIL import Image
from pathlib import Path
import glob
import re
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation as mad

from .models import (
    Pattern,
    GaussianFitWithOffset,
    GaussianFitWithOffset2D
)
from .curve_fitting import (
    fit_gaussian_with_offset,
    fit_gaussian_with_offset_2D
)


month_dict: Dict[int, str] = {
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


def get_next_json_metadata_path(
    root: str,
    year: int,
    month: int,
    day: int,
    n_state: int,
    prefix: str,
) -> Path:
    rootpath = Path(root)
    day_filled: str = str(day).zfill(2)
    year_filled: str = str(year)[-2:]
    month_filled: str = month_dict[month][-3:]
    dirpath = rootpath.joinpath(
        str(year),
        month_dict[month],
        day_filled
    )
    file_no = len(glob.glob(dirpath + '/*.zip'))
    file_filled_start: str = str(file_no).zfill(3)
    file_filled_stop: str = str(file_no+n_state).zfill(3)
    filepath = rootpath.joinpath(
        str(year),
        month_dict[month],
        day_filled,
        f"{prefix}{day_filled}{month_filled}{year_filled}00" +
        f"_{file_filled_start}_{file_filled_stop}_metadata.json"
    )
    return filepath


def get_json_metadata(
    root: str,
    year: int,
    month: int,
    day: int,
    file_start: int,
    file_stop: int,
    prefix: str,
) -> Dict[str, Any]:
    rootpath = Path(root)
    day_filled: str = str(day).zfill(2)
    year_filled: str = str(year)[-2:]
    month_filled: str = month_dict[month][-3:]
    file_filled_start: str = str(file_start).zfill(3)
    file_filled_stop: str = str(file_stop).zfill(3)
    filepath = rootpath.joinpath(
        str(year),
        month_dict[month],
        day_filled,
        f"{prefix}{day_filled}{month_filled}{year_filled}00" +
        f"_{file_filled_start}_{file_filled_stop}_metadata.json"
    )
    with open(filepath, "r") as f:
        meta_data = json.load(f)
    return meta_data


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


def read_frequencies_from_zip(
    archive: ZipFile,
    close: bool = True
) -> Dict[str, Any]:
    frequencies = {}
    for filename in archive.namelist():
        if filename[-17:] == "wavemeterlock.txt":
            with archive.open(filename) as f:
                line = f.readline()
                frequencies = json.loads(line)
    if close:
        archive.close()
    return frequencies


def read_gigatrons_from_zip(
    archive: ZipFile,
    close: bool = True
) -> Dict[str, Any]:
    parameters = {}
    for filename in archive.namelist():
        if filename[-20:] == "gigatrons_params.txt":
            with archive.open(filename) as f:
                line = f.readline()
                parameters = json.loads(line)
    if close:
        archive.close()
    return parameters


def read_ad9959_frequencies_from_zip(
    archive: ZipFile,
    close: bool = True
) -> Dict[str, Any]:
    frequencies = {}
    for filename in archive.namelist():
        if filename[-16:] == "AD9959Config.txt":
            with archive.open(filename) as f:
                line = f.readline()
                _frequencies: Dict[str, List[float, float]] = json.loads(line)
            for key, value in _frequencies.items():
                frequencies[f"ad9959_channel_{key}_freq"] = value[0]
                frequencies[f"ad9959_channel_{key}_amp"] = value[1]
    if close:
        archive.close()
    return frequencies


def read_all_parameter_data_from_zip(
    archive: ZipFile,
    close: bool = True
) -> Dict[str, Any]:
    _all_params = read_parameters_from_zip(archive, False)
    _all_frequencies = read_frequencies_from_zip(archive, False)
    _all_ad9959_frequencies = read_ad9959_frequencies_from_zip(archive)
    _data_dict = _all_params | _all_frequencies
    data_dict = _data_dict | _all_ad9959_frequencies
    return data_dict


def read_time_of_flight_from_zip(
    archive: ZipFile,
    close: bool = True
) -> Tuple[int, np.ndarray]:
    tofs = []
    sampling_rate: int = 0
    sorted_filenames = archive.namelist()
    sorted_filenames.sort(key=natural_keys)
    for filename in sorted_filenames:
        if filename[0:7] == "Tof_PMT":
            with archive.open(filename) as tof_file:
                lines: List[bytes] = tof_file.readlines()
                tofs.append(lines[1:])
                sampling_rate: int = int(
                    lines[0].decode("utf-8").split(",")[0].split(":")[-1]
                )
    if len(tofs) > 1:
        tofs = np.array(tofs, dtype=float).mean(axis=0)
    else:
        tofs = np.array(tofs, dtype=float)
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
        if filename[0:7] == "Tof_PMT":
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


def read_time_of_flight_absorption_from_zip(
    archive: ZipFile,
    close: bool = True
) -> Tuple[int, np.ndarray]:
    tofs = []
    sampling_rate: int = 0
    sorted_filenames = archive.namelist()
    sorted_filenames.sort(key=natural_keys)
    for filename in sorted_filenames:
        if filename[0:7] == "Tof_Abs":
            with archive.open(filename) as tof_file:
                lines: List[bytes] = tof_file.readlines()
                tofs.append(lines[1:])
                sampling_rate: int = int(
                    lines[0].decode("utf-8").split(",")[0].split(":")[-1]
                )
    if len(tofs) > 1:
        tofs = np.array(tofs, dtype=float).mean(axis=0)
    else:
        tofs = np.array(tofs, dtype=float)
    if close:
        archive.close()
    return sampling_rate, tofs


def read_time_of_flight_absorption_from_zip_no_mean(
    archive: ZipFile,
    close: bool = True
) -> Tuple[int, np.ndarray]:
    tofs = []
    sampling_rate: int = 0
    sorted_filenames = archive.namelist()
    sorted_filenames.sort(key=natural_keys)
    for filename in sorted_filenames:
        if filename[0:7] == "Tof_Abs":
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


def convert_chirp_freq_to_tof_bin(
    pmt_distance_in_meter: float,
    chirp_freq_in_MHz: float,
    laser_frequency_in_THz: float,
    sampling_rate: int
) -> int:
    bin_number = pmt_distance_in_meter/(
        chirp_freq_in_MHz*1e6*3e8/(laser_frequency_in_THz*1e12)
    )*sampling_rate
    return int(bin_number)


def convert_blue_mot_detuning_to_aom_frequencies(
    Delta_in_MHz: float,
    delta_a_in_MHz: float,
    delta_b_in_MHz: float,
    delta_R_in_MHz: float,
    R0_aom_in_MHz: float,
    R1p_amp: float,
    B2_amp: float,
    B1m_amp: float,
    R1p_channel_no: int,
    B2_channel_no: int,
    B1m_channel_no: int,
) -> Dict[int, Tuple[float, float]]:
    # CaF structure:
    # level1m = 0.0
    # level1p = 123.0
    level0 = 76.4
    level2 = 148.0
    add9959_freq: Dict = {}
    B1m_in_MHz = (level2 - delta_a_in_MHz)/2
    B2_in_MHz = (
        Delta_in_MHz-level2+level0 +
        2*R0_aom_in_MHz+delta_R_in_MHz+delta_a_in_MHz
    )/2
    R1p_in_MHz = B2_in_MHz - (delta_a_in_MHz-delta_b_in_MHz)/2
    add9959_freq[R1p_channel_no] = (R1p_in_MHz, R1p_amp)
    add9959_freq[B2_channel_no] = (B2_in_MHz, B2_amp)
    add9959_freq[B1m_channel_no] = (B1m_in_MHz, B1m_amp)
    return add9959_freq


def shared_mad_filter_1d(parameters, *observables, threshold=3.5):
    observables_stacked = np.stack(observables, axis=1)
    med = np.median(observables_stacked, axis=0)
    mad_vals = mad(observables_stacked, axis=0, scale='normal')
    z_scores = np.abs((observables_stacked - med) / mad_vals)
    outlier_mask = np.any(z_scores > threshold, axis=1)
    inlier_mask = ~outlier_mask
    filtered = [arr[inlier_mask] for arr in (parameters, *observables)]
    return filtered


def shared_mad_filter_2d(parameter1, parameter2, *observables, threshold=3.5):
    parameter1 = np.asarray(parameter1)
    parameter2 = np.asarray(parameter2)
    observables = tuple(np.asarray(obs) for obs in observables)
    observables_stacked = np.stack(observables, axis=1)
    med = np.median(observables_stacked, axis=0)
    mad_vals = mad(observables_stacked, axis=0, scale='normal')
    mad_vals[mad_vals == 0] = 1e-12
    z_scores = np.abs((observables_stacked - med) / mad_vals)
    outlier_mask = np.any(z_scores > threshold, axis=1)
    inlier_mask = ~outlier_mask
    filtered = [arr[inlier_mask] for arr in (parameter1, parameter2, *observables)]
    return filtered


def groupby1d_bootstrap(
    parameters: np.ndarray,
    values: np.ndarray,
    bootstrap=True,
    n_bootstrap=1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parameters = np.asarray(parameters)
    values = np.asarray(values)
    unique_params, inv_idx = np.unique(parameters, return_inverse=True)
    num_groups = len(unique_params)
    values_mean = np.zeros(num_groups)
    values_err = np.zeros(num_groups)
    all_bootstrap_samples = []
    for i in range(num_groups):
        mask = inv_idx == i
        group_values = values[mask]
        if bootstrap:
            means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(group_values, size=len(group_values), replace=True)
                means.append(np.mean(sample))
            means = np.array(means)
            values_mean[i] = np.mean(means)
            values_err[i] = np.std(means)
            all_bootstrap_samples.append(means)
        else:
            values_mean[i] = np.mean(group_values)
            values_err[i] = np.std(group_values) / np.sqrt(len(group_values))
            all_bootstrap_samples.append(group_values)
    return unique_params, values_mean, values_err, all_bootstrap_samples


def groupby_data_1d(
    parameters: np.ndarray,
    rel_numbers: np.ndarray,
    v_widths: np.ndarray,
    h_widths: np.ndarray,
    v_centres: np.ndarray,
    h_centres: np.ndarray,
    threshold: float = 3.5,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
):
    parameters = np.asarray(parameters)
    rel_numbers = np.asarray(rel_numbers)
    v_widths = np.asarray(v_widths)
    h_widths = np.asarray(h_widths)
    v_centres = np.asarray(v_centres)
    h_centres = np.asarray(h_centres)
    parameters, rel_numbers, v_widths, h_widths, v_centres, h_centres = shared_mad_filter_1d(
        parameters, rel_numbers, v_widths, h_widths, v_centres, h_centres, threshold=threshold
    )
    group = lambda x: groupby1d_bootstrap(parameters, x, bootstrap, n_bootstrap)
    unique_params, number_mean, number_err, _ = group(rel_numbers)
    _, v_widths_mean, v_widths_err, _ = group(v_widths)
    _, h_widths_mean, h_widths_err, _ = group(h_widths)
    _, v_centres_mean, v_centres_err, _ = group(v_centres)
    _, h_centres_mean, h_centres_err, _ = group(h_centres)
    return (
        unique_params, number_mean, number_err,
        v_widths_mean, v_widths_err,
        h_widths_mean, h_widths_err,
        v_centres_mean, v_centres_err,
        h_centres_mean, h_centres_err
    )


def groupby2d_bootstrap(
    parameter1: np.ndarray,
    parameter2: np.ndarray,
    values: np.ndarray,
    bootstrap: bool = True,
    n_bootstrap: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[np.ndarray]]]:
    parameter1 = np.asarray(parameter1)
    parameter2 = np.asarray(parameter2)
    values = np.asarray(values)

    unique_p1 = np.unique(parameter1)
    unique_p2 = np.unique(parameter2)

    num_p1 = len(unique_p1)
    num_p2 = len(unique_p2)

    mean_2d = np.full((num_p1, num_p2), np.nan)
    err_2d = np.full((num_p1, num_p2), np.nan)
    all_bootstrap_samples = [[[] for _ in range(num_p2)] for _ in range(num_p1)]

    for i, p1 in enumerate(unique_p1):
        for j, p2 in enumerate(unique_p2):
            mask = (parameter1 == p1) & (parameter2 == p2)
            group_values = values[mask]

            if group_values.size == 0:
                continue

            if bootstrap:
                means = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(group_values, size=len(group_values), replace=True)
                    means.append(np.mean(sample))
                means = np.array(means)
                mean_2d[i, j] = np.mean(means)
                err_2d[i, j] = np.std(means)
                all_bootstrap_samples[i][j] = means
            else:
                mean_2d[i, j] = np.mean(group_values)
                err_2d[i, j] = np.std(group_values) / np.sqrt(len(group_values))
                all_bootstrap_samples[i][j] = group_values

    return unique_p1, unique_p2, mean_2d, err_2d, all_bootstrap_samples


def groupby_data_2d(
    parameter1: np.ndarray,
    parameter2: np.ndarray,
    rel_numbers: np.ndarray,
    v_widths: np.ndarray,
    h_widths: np.ndarray,
    v_centres: np.ndarray,
    h_centres: np.ndarray,
    threshold: float,
    bootstrap: bool = True,
    n_bootstrap: int = 1000
):
    parameter1, parameter2, rel_numbers, v_widths, h_widths, v_centres, h_centres = shared_mad_filter_2d(
        parameter1, parameter2, rel_numbers, v_widths, h_widths, v_centres, h_centres,
        threshold=threshold
    )
    group = lambda x: groupby2d_bootstrap(parameter1, parameter2, x, bootstrap=bootstrap, n_bootstrap=n_bootstrap)
    unique_p1, unique_p2, number_mean, number_err, _ = group(rel_numbers)
    _, _, v_widths_mean, v_widths_err, _ = group(v_widths)
    _, _, h_widths_mean, h_widths_err, _ = group(h_widths)
    _, _, v_centres_mean, v_centres_err, _ = group(v_centres)
    _, _, h_centres_mean, h_centres_err, _ = group(h_centres)
    return (
        unique_p1, unique_p2, number_mean, number_err,
        v_widths_mean, v_widths_err,
        h_widths_mean, h_widths_err,
        v_centres_mean, v_centres_err,
        h_centres_mean, h_centres_err
    )
