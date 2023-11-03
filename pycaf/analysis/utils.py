from typing import Any, Dict, List, Tuple
from zipfile import ZipFile
from PIL import Image
from pathlib import Path
import re
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline


from .models import (
    Pattern
)
from .curve_fitting import (
    linear,
    fit_linear,
    gaussian_with_offset,
    fit_gaussian_with_offset,
    gaussian_without_offset,
    fit_gaussian_without_offset,
    exponential_without_offset,
    fit_exponential_without_offset,
    exponential_with_offset,
    fit_exponential_with_offset
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
    prefix: str
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
    return ZipFile(filepath)


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
    archive: ZipFile
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
    return channels


def read_analog_patterns_from_zip(
    archive: ZipFile
) -> Dict[str, Pattern]:
    parameters = read_parameters_from_zip(archive)
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
    return channels


def read_images_from_zip(
    archive: ZipFile
) -> List[np.ndarray]:
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
    return images


def read_parameters_from_zip(
    archive: ZipFile
) -> Dict[str, Any]:
    parameters = {}
    for filename in archive.namelist():
        if filename[-14:] == "parameters.txt":
            with archive.open(filename) as parameter_file:
                script_parameters = parameter_file.readlines()
                for line in script_parameters:
                    name, value, _ = line.split(b"\t")
                    parameters[name.decode("utf-8")] = np.float(value)
        elif filename[-18:] == "hardwareReport.txt":
            with archive.open(filename) as hardware_file:
                hardware_parameters = hardware_file.readlines()
                for line in hardware_parameters:
                    name, value, _ = line.split(b"\t")
                    if value.isdigit():
                        parameters[name.decode("utf-8")] = np.float(value)
    return parameters


def read_time_of_flight_from_zip(
    archive: ZipFile
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
    return sampling_rate, tofs


def read_time_of_flight_from_zip_no_mean(
    archive: ZipFile
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
    return sampling_rate, tofs


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
    photon = (count*full_well_capacity*eta_q)/(2**bits_per_channel-1)
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
) -> Tuple[float, float]:
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
    return vertival_fit["sigma"], horizontal_fit["sigma"]


def calculate_cloud_size_from_image_2d_gaussian(
    image: np.ndarray,
    pixel_size: float,
    bin_size: int,
    magnification: float,
) -> Tuple[float, float]:
    # FIXME
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
    return vertival_fit["sigma"], horizontal_fit["sigma"]


def remove_outliers(
        tof: np.ndarray, 
        threshold = 5.0) -> np.ndarray:
    '''Remove outliers in tof and replace it by average of adjacent points.
        Input 1d tof array, output 1d modified array.
        Use lower threshold to apply heavier filtering'''
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

    # Find the indices of the data points with Modified Z-score above the threshold (outliers)
    outliers_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
    normal_indeces = np.arange(0,length)
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

def bin_tof(
        tof: np.ndarray, 
        sample_rate: int, 
        bin_size: float):
    """ Bin tof in time window of bin_size.
        Input a 1d array tof, int sampling_rate and float bin_size in ms.
        Output a 1d array of bined_tofs, a 1d array of bined_times."""
    bined_times = []
    bined_tof = []
    tof_modified = remove_outliers(tof)
    
    # time span in ms:
    time_span = 1000 * len(tof)/sample_rate
    
    # length of output array:
    bin_num = int(time_span / bin_size)

    # number of binned data points
    bin_data_points = int(len(tof)/bin_num)

    
    for i in range(bin_num):
        bined_tof.append((sum(tof_modified[i * bin_data_points : bin_data_points + i * bin_data_points])))
        
    
    bined_times = np.linspace(0, time_span, bin_num)
    
    return bined_tof, bined_times

def bin_tofs(
        tofs: np.ndarray, 
        sample_rate: int, 
        bin_size: float):
    """ Bin tofs in time window of bin_size.
        Input a 2d array tofs, int sampling_rate and float bin_size in ms.
        Output a 2d array of bined_tofs, a 1d array of bined_times."""
    bined_tofs = []
    bined_times = []
    
    # time span in ms:
    time_span = 1000 * len(tofs[0])/sample_rate
    
    # length of output array:
    bin_num = int(time_span / bin_size)

    # number of binned data points
    bin_data_points = int(len(tofs[0])/bin_num)

    for tof in tofs:
        tof_modified = remove_outliers(tof)
        bined_tof = []
        for i in range(bin_num):
            bined_tof.append(sum(tof_modified[i * bin_data_points : bin_data_points + i * bin_data_points]))
        
        bined_tofs.append(bined_tof)
    
    bined_times = np.linspace(0, time_span, bin_num)
    
    return bined_tofs, bined_times

def gaussian_function(x, x0, a, sigma, b):
    return abs(a) * np.exp( - (x - x0) ** 2 / ( 2 * sigma ** 2 )) + b

def triple_gaussian_function(x, x01, x02, x03, a1, a2, a3, sigma1, sigma2, sigma3, b):
    return gaussian_function(x, x01, a1, sigma1, b) + gaussian_function(x, x02, a2, sigma2, 0) + gaussian_function(x, x03, a3, sigma3, 0)


def get_tof_spectrum(tofs, detunings, bined_times, angle = 45.0, threshold = 0.2, velocity_conversion = 0.607, travel_distance = 1400.0, show_images = False):
    """ Get spectrum for each time window, fit into single Gaussian, get velocity for each time window.
        Input 2d array of tofs, bined times and laser detunings in MHz.
        Output transposed tofs_t, first axis time of flight in ms, second axis detuning in MHz.
        And an array of velocities"""
    tofs_t = np.transpose(tofs)
    is_fits = []
    times = []
    velocities = []
    velocity_errors = []
    index = 0
    velocity_conversion = velocity_conversion / np.cos(np.pi*angle/180.0)
    fit_bounds = ((-np.inf, -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,10.0,10.0,10.0,0.0),(0,0,0,np.inf,np.inf,np.inf,30.0,30.0,30.0,np.inf))
    for tof in tofs_t:
        is_fit = True
        x_guess = detunings[np.argmax(tof)]
        a_guess = np.max(tof)
        b_guess = np.min(tof)

        # If no obvious peaks, mark as fit fail
        if is_fit:
            if a_guess - b_guess < threshold * b_guess:
                is_fit = False
        
        # If does't fit, mark as fit fail
        if is_fit:
            try:
                popt, pcov = curve_fit(triple_gaussian_function, detunings, tof, [x_guess, x_guess - 78, x_guess + 78, a_guess, 0.6 * a_guess, 0.5 * a_guess, 20, 20, 20, b_guess], bounds=fit_bounds)
            except RuntimeError:
                is_fit = False
            except ValueError:
                is_fit = False

        # If fitting error is too big, mark as fit fail
        if is_fit:
            fit_error = np.sqrt(np.diag(pcov))[0]
            if abs(fit_error) > 20.0:
                is_fit = False

        # If sidebands are too far away, mark as fit fail
        if is_fit:
            if abs(popt[0] - popt[1]) > 100.0 or abs(popt[0] - popt[2]) > 100.0:
                is_fit = False
        
        # If sidebands are too close, mark as fit fail
        if is_fit:
            if abs(popt[0] - popt[1]) < 50.0 or abs(popt[0] - popt[2]) < 50.0:
                is_fit = False

        # If sidebands order are wrong, mark as fit fail
        if is_fit:
            if (popt[0] - popt[1]) * (popt[0] - popt[2]) > 0:
                is_fit = False

        if is_fit:
            if (popt[3] - popt[4]) < 0 or (popt[3] - popt[5]) < 0:
                is_fit = False

        # If too noisy, mark as fit fail
        if is_fit:
            if np.sqrt(np.diag(pcov))[-1] > np.sqrt(np.diag(pcov))[4]:
                is_fit = False


        # Plot and process only spectrums that fit
        if is_fit:
            if show_images:
                plt.figure()
                plt.plot(detunings, tof, 'ob', label = "data")
                plt.plot(detunings, triple_gaussian_function(detunings, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9]), 'r', label = "fit")
                plt.title("Center detuning: " + str(popt[0]) + "MHz, time window " + str(bined_times[index]) + " ms")
                plt.legend()
            velocities.append( - velocity_conversion * popt[0])
            velocity_errors.append(abs(velocity_conversion * np.sqrt(np.diag(pcov))[0]))
            times.append(bined_times[index])
        index += 1
        
    # Plot velocity vs time of arrival for each bined time windows and interpolated function
    plot_times = np.linspace(min(times) - 1.0, max(times) + 1.0, 100)
    cs = CubicSpline(times, velocities)
    plt.figure()
    plt.errorbar(times, velocities, velocity_errors, fmt = 'ok', label = 'data')
    plt.plot(plot_times, cs(plot_times), 'r', label = 'interpolated curve')
    plt.plot(plot_times, [travel_distance / t for t in plot_times], 'b--', label = "L/t")
    plt.title("Time to velocity conversion")
    plt.xlabel('Time of arrival (ms)')
    plt.ylabel('Velocity (m/s)')
    plt.ylim((0, 500.0))
    plt.legend()

    return cs, velocities, velocity_errors, times, is_fits

def convert_tof_to_velocity_distribution(bined_tof_90_degree, tof_times, velocities, times, cs, background_ratio = 0.1):
    ''' Convert 90 degree tof into velocity distribution.
        Return 1d array of velocities and populations.'''
    
    populations = []
    plot_velocities = []

    # Take the average of background_ratio * length as background
    length = len(bined_tof_90_degree)
    number_of_background_point = (int)(length * background_ratio)
    background = np.average(np.partition(bined_tof_90_degree, number_of_background_point)[:number_of_background_point])

    for i in range(len(tof_times)):
        if(tof_times[i] >= min(times) and tof_times[i] <= max(times)):
            population = bined_tof_90_degree[i] - background
            dv_over_dt = abs(cs(tof_times[i], 1))
            populations.append(population / dv_over_dt)
            plot_velocities.append(cs(tof_times[i]))

    return plot_velocities, populations

def get_velocity_distrubution(
        tof_90_degree: np.ndarray, 
        tofs_angled: np.ndarray, 
        sample_rate: int, 
        bin_size: float, 
        detunings: np.ndarray, 
        angle = 45.0, 
        threshold = 0.2, 
        velocity_conversion = 0.607, 
        background_ratio = 0.1, 
        travel_distance = 1400.0, 
        show_images = False):
    ''' A function to get velocity distribution from 90 degree tof and angled tofs
        Velocity to time relation is plotted from angled tofs
        Return popilation distribution in velocity converting from 90 degree tof
        Inputs: 
        tof_90_degree, a 1d array
        tofs_angled, a 2d array
        sample_rate, in Hz, 
        bin_size, in ms, 
        detunings, in MHz corresponding to axis 0 of tofs_angled, 
        angle, your counterpropogating probe beam angle in degree,
        threshold, any spectrum with (max - min)/min < threshold will be discarded
        velocity_conversion, the Doppler shift without angle in m/s/MHz
        background_ratio, the smallest (background_ratio * length) of tof points will be taken as background and subtracted
        travel_distance, distance between source to porbe in mm
        show_images, if show the spectrum or not
        Returns: 
        1d array holding velocities
        1d array holding populations'''

    # Bin 90 degree tof
    bined_tof_contrl, bined_times = bin_tof(tof_90_degree, sample_rate, bin_size)
    # Bin angled tofs
    bined_tofs, bined_times = bin_tofs(tofs_angled, sample_rate, bin_size)
    # Transpose tofs into velocity spectrum, plot velocity to time relation
    cs, velocities, velocity_errors, times, is_fits = get_tof_spectrum(bined_tofs, detunings, bined_times, angle, threshold, velocity_conversion, travel_distance, show_images)
    # Convert 90 degree tof into velocity distribution
    plot_velocities, populations = convert_tof_to_velocity_distribution(bined_tof_contrl, bined_times, velocities, times, cs, background_ratio)
    
    # Sort plot_velocities
    sorted_indices = np.argsort(plot_velocities)
    sorted_velocities = [plot_velocities[i] for i in sorted_indices]
    sorted_populations = [populations[i] for i in sorted_indices]

    # Plot
    plt.figure()
    plt.plot(sorted_velocities, sorted_populations)
    plt.title("Velocity distribution")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Population (a.u.)")
    plt.xlim((sorted_velocities[0], 200))

    return sorted_velocities, sorted_populations