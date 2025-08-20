from typing import Tuple, List, Union, Dict, Callable
from typing_extensions import Self
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import json
import scipy.constants as cn

from ..analysis.models import (
    Fit,
    LinearFit,
    GaussianFitWithOffset
)
from ..analysis import (
    get_zip_archive,
    read_images_from_zip,
    read_all_parameter_data_from_zip,
    calculate_cloud_size_from_image_1d_gaussian,
    calculate_cloud_size_from_image_2d_gaussian,
    fit_linear,
    fit_quadratic_without_slope,
    fit_exponential_without_offset,
    fit_exponential_with_offset,
    fit_gaussian_with_offset,
    fit_gaussian_without_offset,
    fit_lorentzian_with_offset,
    fit_lorentzian_without_offset,
    fit_trap_frequency_oscillation,
    groupby_data_1d,
    groupby_data_2d
)

from .probeV2 import ProbeV2


fitting_func_map = \
    {
        "linear": fit_linear,
        "quadratic_without_slope": fit_quadratic_without_slope,
        "exponential_without_offset": fit_exponential_without_offset,
        "exponential_with_offset": fit_exponential_with_offset,
        "gaussian_without_offset": fit_gaussian_without_offset,
        "gaussian_with_offset": fit_gaussian_with_offset,
        "lorentzian_with_offset": fit_lorentzian_with_offset,
        "lorentzian_without_offset": fit_lorentzian_without_offset,
        "trap_frequency_oscillation": fit_trap_frequency_oscillation
    }


horizontal_width_aliases = [
    "h_w", "h_width",
    "horizontal_width", "horiz_width"
]
vertical_width_aliases = [
    "v_w", "v_width",
    "vertical_width", "vert_width"
]
horizontal_centre_aliases = [
    "h_p", "h_position",
    "horizontal_position", "horiz_pos"
]
vertical_centre_aliases = [
    "v_p", "v_position",
    "vertical_position", "vert_pos"
]
number_aliases = ["number", "n", "N"]
density_aliases = ["density", "d", "D"]


class ProbeV3(ProbeV2):
    def __init__(
        self,
        config_path: str,
        year: int,
        month: int,
        day: int,
    ) -> None:
        super().__init__(config_path, year, month, day)
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.prefix = self.config["data_prefix"]
        self.rootpath = self.config["data_root_path"]
        self.constants = self.config["constants"]
        self.photon = self.constants["full_well_capacity"]/(
            (2**self.constants["bits_per_channel"]-1)*self.constants["eta_q"]
        )
        self.scale_multiplier = self.constants["magnification"] * \
            self.constants["pixel_size"]*self.constants["binning"]
        self.exposure_time_param = \
            self.constants["cs_exposure_time_parameter"]
        self.camera_trigger_channel = \
            self.constants["cs_camera_trigger_channel_name"]
        self.gamma = self.constants["gamma"]
        self.collection_solid_angle = self.constants["collection_solid_angle"]
        self.magnification = self.constants["magnification"]
        self.pixel_size = self.constants["pixel_size"]
        self.binning = self.constants["binning"]
        self.mass = self.constants["mass"]
        self.year = year
        self.month = month
        self.day = day

    def reset(self) -> Self:
        self.file_start: int = None
        self.file_stop: int = None
        self.display_xlim: Tuple[float, float] = None
        self.display_number_ylim: Tuple[float, float] = None
        self.display_horizontal_width_ylim: Tuple[float, float] = None
        self.display_vertical_width_ylim: Tuple[float, float] = None
        self.display_horizontal_centre_ylim: Tuple[float, float] = None
        self.display_vertical_centre_ylim: Tuple[float, float] = None
        self.display_density_ylim: Tuple[float, float] = None
        self.raw_images: np.ndarray = []
        self.processed_images: np.ndarray = []
        self.h_fits: List[GaussianFitWithOffset] = []
        self.v_fits: List[GaussianFitWithOffset] = []
        self.tofs: np.ndarray = None
        self.tof_sampling_rate: int = None
        self.horizontal_temperature: LinearFit = None
        self.vertical_temperature: LinearFit = None
        self.figsize: Tuple[int, int] = (8, 5)
        self.fmt: str = "ok"
        self.xoffset = 0.0
        self.yoffset = 0.0
        self.xscale = 1.0
        self.yscale = 1.0
        self.xlabel: str = None
        self.ylabel: str = None
        self.title: str = None
        self.row_start = 0
        self.row_end = -1
        self.col_start = 0
        self.col_end = -1
        self.unique_params1: np.ndarray = []
        self.unique_params2: np.ndarray = []
        self.number_mean: np.ndarray = []
        self.number_err: np.ndarray = []
        self.v_widths_mean: np.ndarray = []
        self.v_widths_err: np.ndarray = []
        self.h_widths_mean: np.ndarray = []
        self.h_widths_err: np.ndarray = []
        self.v_centres_mean: np.ndarray = []
        self.v_centres_err: np.ndarray = []
        self.h_centres_mean: np.ndarray = []
        self.h_centres_err: np.ndarray = []
        self.density: np.ndarray = []
        self.density_err: np.ndarray = []
        self.number_fit: Fit = None
        self.h_widths_fit: Fit = None
        self.v_widths_fit: Fit = None
        self.h_centres_fit: Fit = None
        self.v_centres_fit: Fit = None
        self.density_fit: Fit = None
        return self

    def __call__(
        self,
        file_start: int,
        file_stop: int,
        parameters: List[str],
        only_number: bool = False,
        row_start: int = 0,
        row_end: int = -1,
        col_start: int = 0,
        col_end: int = -1,
        discard_runs_upto: int = 0,
        is_bg_included: bool = False,
        first_file_normalization: bool = False,
        fit_2d: bool = False,
        bg_file_start: int = None,
        bg_file_stop: int = None,
        threshold=3.5,
        bootstrap=True,
        n_bootstrap=1000
    ) -> Self:
        self.reset()
        self.file_start = file_start
        self.file_stop = file_stop
        self.parameters = parameters
        self.only_number = only_number
        self.row_start = row_start
        self.row_end = row_end
        self.col_start = col_start
        self.col_end = col_end
        self.discard_runs_upto = discard_runs_upto
        self.threshold = threshold
        self.bootstrap = bootstrap
        self.n_bootstrap = n_bootstrap
        self.bg_file_start = bg_file_start
        self.bg_file_stop = bg_file_stop
        self.fit_2d = fit_2d
        if first_file_normalization:
            if len(parameters) == 1:
                if is_bg_included:
                    self._extract_data_with_bg_included_1d_normalized()
                else:
                    self._extract_data_without_bg_included_1d_normalized()
            elif len(parameters) == 2:
                if is_bg_included:
                    self._extract_data_with_bg_included_2d_normalized()
                else:
                    self._extract_data_without_bg_included_2d_normalized()
        else:
            if len(parameters) == 1:
                if bg_file_start and bg_file_stop:
                    self._extract_data_with_bg_file_1d()
                else:
                    if is_bg_included:
                        self._extract_data_with_bg_included_1d()
                    else:
                        self._extract_data_without_bg_included_1d()
            elif len(parameters) == 2:
                if is_bg_included:
                    self._extract_data_with_bg_included_2d()
                else:
                    self._extract_data_without_bg_included_2d()
        return self
    
    def _extract_data_with_bg_file_1d(self) -> Self:
        rel_numbers, v_widths, h_widths = [], [], []
        v_centres, h_centres, parameters = [], [], []
        raw_images, processed_images = [], []
        bg_images = []
        for bg_fileno in range(self.bg_file_start, self.bg_file_stop+1, 1):
            yag_off_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, bg_fileno, self.prefix
                )
            )
            bg_images.append(yag_off_images)
        n_iter = len(bg_images)
        for k, fileno in enumerate(range(self.file_start, self.file_stop+1, 1)):
            yag_on_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            _images = yag_on_images - bg_images[k%n_iter]
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            raw_img = np.mean(
                _images[self.discard_runs_upto:],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    self.discard_runs_upto:,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            
            _param = _params[self.parameters[0]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            n = number_multiplier*np.nansum(processed_img)
            if not self.only_number:
                if self.fit_2d:
                    fit = calculate_cloud_size_from_image_2d_gaussian(
                        processed_img,
                        pixel_size=self.pixel_size,
                        bin_size=self.binning,
                        magnification=self.magnification
                    )
                    if fit is not None:
                        xs = np.sqrt(fit.xwidth**2*np.cos(fit.theta)**2+fit.ywidth**2*np.sin(fit.theta)**2)
                        ys = np.sqrt(fit.xwidth**2*np.sin(fit.theta)**2+fit.ywidth**2*np.cos(fit.theta)**2)
                        rel_numbers.append(n)
                        parameters.append(_param)
                        raw_images.append(raw_img)
                        processed_images.append(processed_img)
                        self.v_fits.append(fit)
                        self.h_fits.append(fit)
                        v_widths.append(xs)
                        v_centres.append(fit.xcentre)
                        h_widths.append(ys)
                        h_centres.append(fit.ycentre)
                    else:
                        print(f"cloud fitting error in fileno: {fileno}")
                else:
                    v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                        processed_img,
                        pixel_size=self.pixel_size,
                        bin_size=self.binning,
                        magnification=self.magnification
                    )
                    if (v_fit is not None) and (h_fit is not None):
                        rel_numbers.append(n)
                        parameters.append(_param)
                        raw_images.append(raw_img)
                        processed_images.append(processed_img)
                        self.v_fits.append(v_fit)
                        self.h_fits.append(h_fit)
                        v_widths.append(v_fit.width)
                        v_centres.append(v_fit.centre)
                        h_widths.append(h_fit.width)
                        h_centres.append(h_fit.centre)
                    else:
                        print(f"cloud fitting error in fileno: {fileno}")
            else:
                rel_numbers.append(n)
                parameters.append(_param)
                raw_images.append(raw_img)
                processed_images.append(processed_img)
                self.v_fits.append(None)
                self.h_fits.append(None)
                v_widths.append(0.0)
                v_centres.append(0.0)
                h_widths.append(0.0)
                h_centres.append(0.0)
        self.unique_params1, self.number_mean, self.number_err, \
        self.v_widths_mean, self.v_widths_err, \
        self.h_widths_mean, self.h_widths_err, \
        self.v_centres_mean, self.v_centres_err, \
        self.h_centres_mean, self.h_centres_err = \
            groupby_data_1d(
                parameters, rel_numbers, v_widths,
                h_widths, v_centres, h_centres
            )
        self.raw_images = np.array(raw_images)
        self.processed_images = np.array(processed_images)
        return self


    def _extract_data_with_bg_included_1d(self) -> Self:
        rel_numbers, v_widths, h_widths = [], [], []
        v_centres, h_centres, parameters = [], [], []
        raw_images, processed_images = [], []
        for fileno in range(self.file_start, self.file_stop+1):
            _images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            raw_img = np.mean(
                _images[0+self.discard_runs_upto::2] - 
                _images[1+self.discard_runs_upto::2],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    0+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ] - 
                _images[
                    1+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            
            _param = _params[self.parameters[0]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            n = number_multiplier*np.nansum(processed_img)

            if not self.only_number:
                if self.fit_2d:
                    fit = calculate_cloud_size_from_image_2d_gaussian(
                        processed_img,
                        pixel_size=self.pixel_size,
                        bin_size=self.binning,
                        magnification=self.magnification
                    )
                    if fit is not None:
                        xs = np.sqrt(fit.xwidth**2*np.cos(fit.theta)**2+fit.ywidth**2*np.sin(fit.theta)**2)
                        ys = np.sqrt(fit.xwidth**2*np.sin(fit.theta)**2+fit.ywidth**2*np.cos(fit.theta)**2)
                        rel_numbers.append(n)
                        parameters.append(_param)
                        raw_images.append(raw_img)
                        processed_images.append(processed_img)
                        self.v_fits.append(fit)
                        self.h_fits.append(fit)
                        v_widths.append(xs)
                        v_centres.append(fit.xcentre)
                        h_widths.append(ys)
                        h_centres.append(fit.ycentre)
                    else:
                        print(f"cloud fitting error in fileno: {fileno}")
                else:
                    v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                        processed_img,
                        pixel_size=self.pixel_size,
                        bin_size=self.binning,
                        magnification=self.magnification
                    )
                    if (v_fit is not None) and (h_fit is not None):
                        rel_numbers.append(n)
                        parameters.append(_param)
                        raw_images.append(raw_img)
                        processed_images.append(processed_img)
                        self.v_fits.append(v_fit)
                        self.h_fits.append(h_fit)
                        v_widths.append(v_fit.width)
                        v_centres.append(v_fit.centre)
                        h_widths.append(h_fit.width)
                        h_centres.append(h_fit.centre)
                    else:
                        print(f"cloud fitting error in fileno: {fileno}")
            else:
                rel_numbers.append(n)
                parameters.append(_param)
                raw_images.append(raw_img)
                processed_images.append(processed_img)
                self.v_fits.append(None)
                self.h_fits.append(None)
                v_widths.append(0.0)
                v_centres.append(0.0)
                h_widths.append(0.0)
                h_centres.append(0.0)
        self.unique_params1, self.number_mean, self.number_err, \
        self.v_widths_mean, self.v_widths_err, \
        self.h_widths_mean, self.h_widths_err, \
        self.v_centres_mean, self.v_centres_err, \
        self.h_centres_mean, self.h_centres_err = \
            groupby_data_1d(
                parameters, rel_numbers, v_widths,
                h_widths, v_centres, h_centres
            )
        self.raw_images = np.array(raw_images)
        self.processed_images = np.array(processed_images)
        return self
    
    def _extract_data_with_bg_included_1d_2d_fit(self) -> Self:
        rel_numbers, v_widths, h_widths = [], [], []
        v_centres, h_centres, parameters = [], [], []
        raw_images, processed_images = [], []
        for fileno in range(self.file_start, self.file_stop+1):
            _images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            raw_img = np.mean(
                _images[0+self.discard_runs_upto::2] - 
                _images[1+self.discard_runs_upto::2],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    0+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ] - 
                _images[
                    1+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            
            _param = _params[self.parameters[0]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            n = number_multiplier*np.nansum(processed_img)

            if not self.only_number:
                if self.fit_2d:
                    fit = calculate_cloud_size_from_image_2d_gaussian(
                        processed_img,
                        pixel_size=self.pixel_size,
                        bin_size=self.binning,
                        magnification=self.magnification
                    )
                    if fit is not None:
                        xs = np.sqrt(fit.xwidth**2*np.cos(fit.theta)**2+fit.ywidth**2*np.sin(fit.theta)**2)
                        ys = np.sqrt(fit.xwidth**2*np.sin(fit.theta)**2+fit.ywidth**2*np.cos(fit.theta)**2)
                        rel_numbers.append(n)
                        parameters.append(_param)
                        raw_images.append(raw_img)
                        processed_images.append(processed_img)
                        self.v_fits.append(fit)
                        self.h_fits.append(fit)
                        v_widths.append(xs)
                        v_centres.append(fit.xcentre)
                        h_widths.append(ys)
                        h_centres.append(fit.ycentre)
                    else:
                        print(f"cloud fitting error in fileno: {fileno}")
                else:
                    v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                        processed_img,
                        pixel_size=self.pixel_size,
                        bin_size=self.binning,
                        magnification=self.magnification
                    )
                    if (v_fit is not None) and (h_fit is not None):
                        rel_numbers.append(n)
                        parameters.append(_param)
                        raw_images.append(raw_img)
                        processed_images.append(processed_img)
                        self.v_fits.append(v_fit)
                        self.h_fits.append(h_fit)
                        v_widths.append(v_fit.width)
                        v_centres.append(v_fit.centre)
                        h_widths.append(h_fit.width)
                        h_centres.append(h_fit.centre)
                    else:
                        print(f"cloud fitting error in fileno: {fileno}")
            else:
                rel_numbers.append(n)
                parameters.append(_param)
                raw_images.append(raw_img)
                processed_images.append(processed_img)
                self.v_fits.append(None)
                self.h_fits.append(None)
                v_widths.append(0.0)
                v_centres.append(0.0)
                h_widths.append(0.0)
                h_centres.append(0.0)
        self.unique_params1, self.number_mean, self.number_err, \
        self.v_widths_mean, self.v_widths_err, \
        self.h_widths_mean, self.h_widths_err, \
        self.v_centres_mean, self.v_centres_err, \
        self.h_centres_mean, self.h_centres_err = \
            groupby_data_1d(
                parameters, rel_numbers, v_widths,
                h_widths, v_centres, h_centres
            )
        self.raw_images = np.array(raw_images)
        self.processed_images = np.array(processed_images)
        return self
    
    def _extract_data_without_bg_included_1d(self) -> Self:
        rel_numbers, v_widths, h_widths = [], [], []
        v_centres, h_centres, parameters = [], [], []
        raw_images, processed_images = [], []
        for fileno in range(self.file_start, self.file_stop+1, 2):
            yag_on_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            yag_off_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+1, self.prefix
                )
            )
            _images = yag_on_images - yag_off_images
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            raw_img = np.mean(
                _images[self.discard_runs_upto:],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    self.discard_runs_upto:,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            
            _param = _params[self.parameters[0]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            n = number_multiplier*np.nansum(processed_img)
            if not self.only_number:
                if self.fit_2d:
                    fit = calculate_cloud_size_from_image_2d_gaussian(
                        processed_img,
                        pixel_size=self.pixel_size,
                        bin_size=self.binning,
                        magnification=self.magnification
                    )
                    if fit is not None:
                        xs = np.sqrt(fit.xwidth**2*np.cos(fit.theta)**2+fit.ywidth**2*np.sin(fit.theta)**2)
                        ys = np.sqrt(fit.xwidth**2*np.sin(fit.theta)**2+fit.ywidth**2*np.cos(fit.theta)**2)
                        rel_numbers.append(n)
                        parameters.append(_param)
                        raw_images.append(raw_img)
                        processed_images.append(processed_img)
                        self.v_fits.append(fit)
                        self.h_fits.append(fit)
                        v_widths.append(xs)
                        v_centres.append(fit.xcentre)
                        h_widths.append(ys)
                        h_centres.append(fit.ycentre)
                    else:
                        print(f"cloud fitting error in fileno: {fileno}")
                else:
                    v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                        processed_img,
                        pixel_size=self.pixel_size,
                        bin_size=self.binning,
                        magnification=self.magnification
                    )
                    if (v_fit is not None) and (h_fit is not None):
                        rel_numbers.append(n)
                        parameters.append(_param)
                        raw_images.append(raw_img)
                        processed_images.append(processed_img)
                        self.v_fits.append(v_fit)
                        self.h_fits.append(h_fit)
                        v_widths.append(v_fit.width)
                        v_centres.append(v_fit.centre)
                        h_widths.append(h_fit.width)
                        h_centres.append(h_fit.centre)
                    else:
                        print(f"cloud fitting error in fileno: {fileno}")
            else:
                rel_numbers.append(n)
                parameters.append(_param)
                raw_images.append(raw_img)
                processed_images.append(processed_img)
                self.v_fits.append(None)
                self.h_fits.append(None)
                v_widths.append(0.0)
                v_centres.append(0.0)
                h_widths.append(0.0)
                h_centres.append(0.0)
        self.unique_params1, self.number_mean, self.number_err, \
        self.v_widths_mean, self.v_widths_err, \
        self.h_widths_mean, self.h_widths_err, \
        self.v_centres_mean, self.v_centres_err, \
        self.h_centres_mean, self.h_centres_err = \
            groupby_data_1d(
                parameters, rel_numbers, v_widths,
                h_widths, v_centres, h_centres
            )
        self.raw_images = np.array(raw_images)
        self.processed_images = np.array(processed_images)
        return self

    def _extract_data_with_bg_included_2d(self) -> Self:
        rel_numbers, v_widths, h_widths = [], [], []
        v_centres, h_centres, parameters1, parameters2 = [], [], [], []
        raw_images, processed_images = [], []
        for fileno in range(self.file_start, self.file_stop+1):
            _images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            raw_img = np.mean(
                _images[0+self.discard_runs_upto::2] - 
                _images[1+self.discard_runs_upto::2],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    0+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ] - 
                _images[
                    1+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            
            _param1 = _params[self.parameters[0]]
            _param2 = _params[self.parameters[1]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            n = number_multiplier*np.nansum(processed_img)

            if not self.only_number:
                v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                    processed_img,
                    pixel_size=self.pixel_size,
                    bin_size=self.binning,
                    magnification=self.magnification
                )
                if (v_fit is not None) and (h_fit is not None):
                    v_widths.append(v_fit.width)
                    v_centres.append(v_fit.centre)
                    h_widths.append(h_fit.width)
                    h_centres.append(h_fit.centre)
                    rel_numbers.append(n)
                    parameters1.append(_param1)
                    parameters2.append(_param2)
                    raw_images.append(raw_img)
                    processed_images.append(processed_img)
                    self.v_fits.append(v_fit)
                    self.h_fits.append(h_fit)
            else:
                v_widths.append(0.0)
                v_centres.append(0.0)
                h_widths.append(0.0)
                h_centres.append(0.0)
                rel_numbers.append(n)
                parameters1.append(_param1)
                parameters2.append(_param2)
                raw_images.append(raw_img)
                processed_images.append(processed_img)
                self.v_fits.append(None)
                self.h_fits.append(None)
        self.unique_params1, self.unique_params2, \
        self.number_mean, self.number_err, \
        self.v_widths_mean, self.v_widths_err, \
        self.h_widths_mean, self.h_widths_err, \
        self.v_centres_mean, self.v_centres_err, \
        self.h_centres_mean, self.h_centres_err = \
            groupby_data_2d(
                parameters1, parameters2, rel_numbers, 
                v_widths, h_widths, v_centres, h_centres,
                self.threshold
            )
        self.raw_images = np.array(raw_images)
        self.processed_images = np.array(processed_images)
        return self
    
    def _extract_data_without_bg_included_2d(self) -> Self:
        rel_numbers, v_widths, h_widths = [], [], []
        v_centres, h_centres, parameters1, parameters2 = [], [], [], []
        raw_images, processed_images = [], []
        for fileno in range(self.file_start, self.file_stop+1, 2):
            yag_on_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            yag_off_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+1, self.prefix
                )
            )
            _images = yag_on_images - yag_off_images
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            raw_img = np.mean(
                _images[self.discard_runs_upto:],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    self.discard_runs_upto:,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            
            _param1 = _params[self.parameters[0]]
            _param2 = _params[self.parameters[1]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            n = number_multiplier*np.nansum(processed_img)
            if not self.only_number:
                v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                    processed_img,
                    pixel_size=self.pixel_size,
                    bin_size=self.binning,
                    magnification=self.magnification
                )
                if (v_fit is not None) and (h_fit is not None):
                    v_widths.append(v_fit.width)
                    v_centres.append(v_fit.centre)
                    h_widths.append(h_fit.width)
                    h_centres.append(h_fit.centre)
                    rel_numbers.append(n)
                    parameters1.append(_param1)
                    parameters2.append(_param2)
                    raw_images.append(raw_img)
                    processed_images.append(processed_img)
                    self.v_fits.append(v_fit)
                    self.h_fits.append(h_fit)
            else:
                v_widths.append(0.0)
                v_centres.append(0.0)
                h_widths.append(0.0)
                h_centres.append(0.0)
                rel_numbers.append(n)
                parameters1.append(_param1)
                parameters2.append(_param2)
                raw_images.append(raw_img)
                processed_images.append(processed_img)
                self.v_fits.append(None)
                self.h_fits.append(None)
        self.unique_params1, self.unique_params2, \
        self.number_mean, self.number_err, \
        self.v_widths_mean, self.v_widths_err, \
        self.h_widths_mean, self.h_widths_err, \
        self.v_centres_mean, self.v_centres_err, \
        self.h_centres_mean, self.h_centres_err = \
            groupby_data_2d(
                parameters1, parameters2, rel_numbers, 
                v_widths, h_widths, v_centres, h_centres,
                self.threshold
            )
        self.raw_images = np.array(raw_images)
        self.processed_images = np.array(processed_images)
        return self

    def _extract_data_with_bg_included_1d_normalized(self) -> Self:
        rel_numbers, v_widths, h_widths = [], [], []
        v_centres, h_centres, parameters = [], [], []
        raw_images, processed_images = [], []
        for fileno in range(self.file_start, self.file_stop+1, 2):
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            _param = _params[self.parameters[0]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            _images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            raw_img = np.mean(
                _images[0+self.discard_runs_upto::2] - 
                _images[1+self.discard_runs_upto::2],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    0+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ] - 
                _images[
                    1+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            n1 = number_multiplier*np.nansum(processed_img)
            
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+1, self.prefix
                )
            )
            _param = _params[self.parameters[0]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            _images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+1, self.prefix
                )
            )
            raw_img = np.mean(
                _images[0+self.discard_runs_upto::2] - 
                _images[1+self.discard_runs_upto::2],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    0+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ] - 
                _images[
                    1+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            n2 = number_multiplier*np.nansum(processed_img)

            n = n2/n1

            if not self.only_number:
                v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                    processed_img,
                    pixel_size=self.pixel_size,
                    bin_size=self.binning,
                    magnification=self.magnification
                )
                if (v_fit is not None) and (h_fit is not None):
                    rel_numbers.append(n)
                    parameters.append(_param)
                    raw_images.append(raw_img)
                    processed_images.append(processed_img)
                    self.v_fits.append(v_fit)
                    self.h_fits.append(h_fit)
                    v_widths.append(v_fit.width)
                    v_centres.append(v_fit.centre)
                    h_widths.append(h_fit.width)
                    h_centres.append(h_fit.centre)
            else:
                rel_numbers.append(n)
                parameters.append(_param)
                raw_images.append(raw_img)
                processed_images.append(processed_img)
                self.v_fits.append(None)
                self.h_fits.append(None)
                v_widths.append(0.0)
                v_centres.append(0.0)
                h_widths.append(0.0)
                h_centres.append(0.0)
        self.unique_params1, self.number_mean, self.number_err, \
        self.v_widths_mean, self.v_widths_err, \
        self.h_widths_mean, self.h_widths_err, \
        self.v_centres_mean, self.v_centres_err, \
        self.h_centres_mean, self.h_centres_err = \
            groupby_data_1d(
                parameters, rel_numbers, v_widths,
                h_widths, v_centres, h_centres
            )
        self.raw_images = np.array(raw_images)
        self.processed_images = np.array(processed_images)
        return self

    def _extract_data_without_bg_included_1d_normalized(self) -> Self:
        rel_numbers, v_widths, h_widths = [], [], []
        v_centres, h_centres, parameters = [], [], []
        raw_images, processed_images = [], []
        for fileno in range(self.file_start, self.file_stop+1, 4):
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            _param = _params[self.parameters[0]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            yag_on_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            yag_off_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+2, self.prefix
                )
            )
            _images = yag_on_images - yag_off_images
            raw_img = np.mean(
                _images[self.discard_runs_upto:],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    self.discard_runs_upto:,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            n1 = number_multiplier*np.nansum(processed_img)

            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+2, self.prefix
                )
            )
            _param = _params[self.parameters[0]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            yag_on_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+1, self.prefix
                )
            )
            yag_off_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+3, self.prefix
                )
            )
            _images = yag_on_images - yag_off_images
            raw_img = np.mean(
                _images[self.discard_runs_upto:],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    self.discard_runs_upto:,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            n2 = number_multiplier*np.nansum(processed_img)

            n = n2/n1

            if not self.only_number:
                v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                    processed_img,
                    pixel_size=self.pixel_size,
                    bin_size=self.binning,
                    magnification=self.magnification
                )
                if (v_fit is not None) and (h_fit is not None):
                    rel_numbers.append(n)
                    parameters.append(_param)
                    raw_images.append(raw_img)
                    processed_images.append(processed_img)
                    self.v_fits.append(v_fit)
                    self.h_fits.append(h_fit)
                    v_widths.append(v_fit.width)
                    v_centres.append(v_fit.centre)
                    h_widths.append(h_fit.width)
                    h_centres.append(h_fit.centre)
            else:
                rel_numbers.append(n)
                parameters.append(_param)
                raw_images.append(raw_img)
                processed_images.append(processed_img)
                self.v_fits.append(None)
                self.h_fits.append(None)
                v_widths.append(0.0)
                v_centres.append(0.0)
                h_widths.append(0.0)
                h_centres.append(0.0)
        self.unique_params1, self.number_mean, self.number_err, \
        self.v_widths_mean, self.v_widths_err, \
        self.h_widths_mean, self.h_widths_err, \
        self.v_centres_mean, self.v_centres_err, \
        self.h_centres_mean, self.h_centres_err = \
            groupby_data_1d(
                parameters, rel_numbers, v_widths,
                h_widths, v_centres, h_centres
            )
        self.raw_images = np.array(raw_images)
        self.processed_images = np.array(processed_images)
        return self

    def _extract_data_with_bg_included_2d_normalized(self) -> Self:
        rel_numbers, v_widths, h_widths = [], [], []
        v_centres, h_centres, parameters1, parameters2 = [], [], [], []
        raw_images, processed_images = [], []
        for fileno in range(self.file_start, self.file_stop+1, 2):
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            _images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            raw_img = np.mean(
                _images[0+self.discard_runs_upto::2] - 
                _images[1+self.discard_runs_upto::2],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    0+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ] - 
                _images[
                    1+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            n1 = number_multiplier*np.nansum(processed_img)
            
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+1, self.prefix
                )
            )
            _param1 = _params[self.parameters[0]]
            _param2 = _params[self.parameters[1]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            _images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+1, self.prefix
                )
            )
            raw_img = np.mean(
                _images[0+self.discard_runs_upto::2] - 
                _images[1+self.discard_runs_upto::2],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    0+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ] - 
                _images[
                    1+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            n2 = number_multiplier*np.nansum(processed_img)

            n = n2/n1

            if not self.only_number:
                v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                    processed_img,
                    pixel_size=self.pixel_size,
                    bin_size=self.binning,
                    magnification=self.magnification
                )
                if (v_fit is not None) and (h_fit is not None):
                    v_widths.append(v_fit.width)
                    v_centres.append(v_fit.centre)
                    h_widths.append(h_fit.width)
                    h_centres.append(h_fit.centre)
                    rel_numbers.append(n)
                    parameters1.append(_param1)
                    parameters2.append(_param2)
                    raw_images.append(raw_img)
                    processed_images.append(processed_img)
                    self.v_fits.append(v_fit)
                    self.h_fits.append(h_fit)
            else:
                v_widths.append(0.0)
                v_centres.append(0.0)
                h_widths.append(0.0)
                h_centres.append(0.0)
                rel_numbers.append(n)
                parameters1.append(_param1)
                parameters2.append(_param2)
                raw_images.append(raw_img)
                processed_images.append(processed_img)
                self.v_fits.append(None)
                self.h_fits.append(None)
        self.unique_params1, self.unique_params2, \
        self.number_mean, self.number_err, \
        self.v_widths_mean, self.v_widths_err, \
        self.h_widths_mean, self.h_widths_err, \
        self.v_centres_mean, self.v_centres_err, \
        self.h_centres_mean, self.h_centres_err = \
            groupby_data_2d(
                parameters1, parameters2, rel_numbers, v_widths,
                h_widths, v_centres, h_centres, self.threshold
            )
        self.raw_images = np.array(raw_images)
        self.processed_images = np.array(processed_images)
        return self
    
    def _extract_data_without_bg_included_2d_normalized(self) -> Self:
        rel_numbers, v_widths, h_widths = [], [], []
        v_centres, h_centres, parameters1, parameters2 = [], [], [], []
        raw_images, processed_images = [], []
        for fileno in range(self.file_start, self.file_stop+1, 4):
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            yag_on_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno, self.prefix
                )
            )
            yag_off_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+2, self.prefix
                )
            )
            _images = yag_on_images - yag_off_images
            raw_img = np.mean(
                _images[self.discard_runs_upto:],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    self.discard_runs_upto:,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            n1 = number_multiplier*np.nansum(processed_img)
            
            _params = read_all_parameter_data_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+1, self.prefix
                )
            )
            _param1 = _params[self.parameters[0]]
            _param2 = _params[self.parameters[1]]
            exposure_time = _params[self.exposure_time_param]*1e-5
            number_multiplier = self.photon/(
                exposure_time*self.gamma*self.collection_solid_angle
            )
            yag_on_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+1, self.prefix
                )
            )
            yag_off_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month,
                    self.day, fileno+3, self.prefix
                )
            )
            _images = yag_on_images - yag_off_images
            raw_img = np.mean(
                _images[0+self.discard_runs_upto::2] - 
                _images[1+self.discard_runs_upto::2],
                axis=0
            )
            processed_img = np.mean(
                _images[
                    0+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ] - 
                _images[
                    1+self.discard_runs_upto::2,
                    self.col_start:self.col_end,
                    self.row_start:self.row_end
                ],
                axis=0
            )
            n2 = number_multiplier*np.nansum(processed_img)

            n = n2/n1

            if not self.only_number:
                v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                    processed_img,
                    pixel_size=self.pixel_size,
                    bin_size=self.binning,
                    magnification=self.magnification
                )
                if (v_fit is not None) and (h_fit is not None):
                    v_widths.append(v_fit.width)
                    v_centres.append(v_fit.centre)
                    h_widths.append(h_fit.width)
                    h_centres.append(h_fit.centre)
                    rel_numbers.append(n)
                    parameters1.append(_param1)
                    parameters2.append(_param2)
                    raw_images.append(raw_img)
                    processed_images.append(processed_img)
                    self.v_fits.append(v_fit)
                    self.h_fits.append(h_fit)
            else:
                v_widths.append(0.0)
                v_centres.append(0.0)
                h_widths.append(0.0)
                h_centres.append(0.0)
                rel_numbers.append(n)
                parameters1.append(_param1)
                parameters2.append(_param2)
                raw_images.append(raw_img)
                processed_images.append(processed_img)
                self.v_fits.append(None)
                self.h_fits.append(None)
        self.unique_params1, self.unique_params2, \
        self.number_mean, self.number_err, \
        self.v_widths_mean, self.v_widths_err, \
        self.h_widths_mean, self.h_widths_err, \
        self.v_centres_mean, self.v_centres_err, \
        self.h_centres_mean, self.h_centres_err = \
            groupby_data_2d(
                parameters1, parameters2, rel_numbers, v_widths,
                h_widths, v_centres, h_centres, self.threshold
            )
        self.raw_images = np.array(raw_images)
        self.processed_images = np.array(processed_images)
        return self

    def exclude_parameters_by_index(
        self,
        indices: List[int]
    ) -> Self:
        self.unique_params1 = np.delete(self.unique_params1, indices, axis=0)
        if len(self.unique_params2):
            self.unique_params2 = np.delete(self.unique_params2, indices, axis=0)
        self.number_mean = np.delete(self.number_mean, indices, axis=0)
        self.number_err = np.delete(self.number_err, indices, axis=0)
        self.v_widths_mean = np.delete(self.v_widths_mean, indices, axis=0)
        self.v_widths_err = np.delete(self.v_widths_err, indices, axis=0)
        self.h_widths_mean = np.delete(self.h_widths_mean, indices, axis=0)
        self.h_widths_err = np.delete(self.h_widths_err, indices, axis=0)
        self.v_centres_mean = np.delete(self.v_centres_mean, indices, axis=0)
        self.v_centres_err = np.delete(self.v_centres_err, indices, axis=0)
        self.h_centres_mean = np.delete(self.h_centres_mean, indices, axis=0)
        self.h_centres_err = np.delete(self.h_centres_err, indices, axis=0)
        return self

    def fit_number_variation(
        self,
        fitting_function: Callable
    ) -> Self:
        self.number_fit: Fit = fitting_function(
            self.unique_params1,
            self.number_mean,
            self.number_err
        )
        return self

    def fit_horizontal_width_variation(
        self,
        fitting_function: Callable
    ) -> Self:
        self.h_widths_fit: Fit = fitting_function(
            self.unique_params1,
            self.h_widths_mean,
            self.h_widths_err
        )
        return self
    
    def fit_vertical_width_variation(
        self,
        fitting_function: Callable
    ) -> Self:
        self.v_widths_fit: Fit = fitting_function(
            self.unique_params1,
            self.v_widths_mean,
            self.v_widths_err
        )
        return self

    def display_variation(self) -> Self:
        if len(self.parameters) == 1:
            self._display_variation_1d()
        elif len(self.parameters) == 2:
            self._display_variation_2d()
        return self

    def _display_variation_1d(self) -> Self:
        list_of_y: List[np.ndarray] = [
            self.number_mean,
            self.h_widths_mean,
            self.v_widths_mean,
            self.h_centres_mean,
            self.v_centres_mean
        ]
        list_of_yerr: List[np.ndarray] = [
            self.number_err,
            self.h_widths_err,
            self.v_widths_err,
            self.h_centres_err,
            self.v_centres_err
        ]
        list_of_ylabels: List[str] = [
            "Molecule Numbers",
            "Vertical width [mm]",
            "Horizontal width [mm]",
            "Vertical Centre [mm]",
            "Horizontal Centre [mm]"
        ]
        list_of_yfactors: List[float] = [
            1.0,
            1e3,
            1e3,
            1e3,
            1e3
        ]
        list_of_yfits: List[Fit] = [
            self.number_fit,
            self.h_widths_fit,
            self.v_widths_fit,
            self.h_centres_fit,
            self.v_centres_fit
        ]
        list_of_ylimits: List[Tuple[float, float]] = [
            self.display_number_ylim,
            self.display_horizontal_width_ylim,
            self.display_vertical_width_ylim,
            self.display_horizontal_centre_ylim,
            self.display_vertical_centre_ylim
        ]
        for y, yerr, ylabel, yfactor, yfit, ylim in zip(
            list_of_y,
            list_of_yerr,
            list_of_ylabels,
            list_of_yfactors,
            list_of_yfits,
            list_of_ylimits
        ):
            if y is not None:
                yerr[yerr < 0] = 0.0
                _, ax = plt.subplots(1, 1, figsize=self.figsize)
                ax.errorbar(
                    self.unique_params1,
                    y*yfactor,
                    yerr=yerr*yfactor,
                    fmt=self.fmt
                )
                ax.set_xlabel(self.xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(self.title)
                ax.set_ylim(ylim)
                ax.set_xlim(self.display_xlim)
                if yfit is not None:
                    ax.plot(
                        yfit.x_fine,
                        yfit.y_fine*yfactor,
                        "-r"
                    )
                    ax.text(
                        1.03, 0.9,
                        "Fitting:"+yfit.func_str+yfit.args_str,
                        transform=ax.transAxes,
                        fontsize=12,
                        verticalalignment='top',
                        bbox=dict(
                            boxstyle='round',
                            facecolor='lightsteelblue',
                            alpha=0.15
                        )
                    )
            if self.only_number:
                break
        return self
    
    def _display_variation_2d(self) -> Self:
        list_of_y: List[np.ndarray] = [
            self.number_mean,
            self.h_widths_mean,
            self.v_widths_mean,
            self.h_centres_mean,
            self.v_centres_mean
        ]
        list_of_yerr: List[np.ndarray] = [
            self.number_err,
            self.h_widths_err,
            self.v_widths_err,
            self.h_centres_err,
            self.v_centres_err
        ]
        list_of_titles: List[str] = [
            "Molecule Numbers",
            "Vertical width [mm]",
            "Horizontal width [mm]",
            "Vertical Centre [mm]",
            "Horizontal Centre [mm]"
        ]
        list_of_yfactors: List[float] = [
            1.0,
            1e3,
            1e3,
            1e3,
            1e3
        ]
        for y, yerr, title, yfactor in zip(
            list_of_y,
            list_of_yerr,
            list_of_titles,
            list_of_yfactors
        ):
            if y is not None:
                yerr[yerr < 0] = 0.0
                fig, ax = plt.subplots(1, 2, figsize=self.figsize)
                fig.subplots_adjust(wspace=0.05)
                im0 = ax[0].imshow(
                    yfactor*y.T,
                    extent=(
                        (self.unique_params1[0]-self.xoffset)*self.xscale,
                        (self.unique_params1[-1]-self.xoffset)*self.xscale,
                        (self.unique_params2[0]-self.yoffset)*self.yscale,
                        (self.unique_params2[-1]-self.yoffset)*self.yscale
                    ),
                    origin="lower"
                )
                im1 = ax[1].imshow(
                    yfactor*yerr.T,
                    extent=(
                        (self.unique_params1[0]-self.xoffset)*self.xscale,
                        (self.unique_params1[-1]-self.xoffset)*self.xscale,
                        (self.unique_params2[0]-self.yoffset)*self.yscale,
                        (self.unique_params2[-1]-self.yoffset)*self.yscale
                    ),
                    origin="lower"
                )
                fig.colorbar(
                    im0, ax=ax[0],
                    location="bottom", shrink=1.0, label=f"Mean: {title}")
                fig.colorbar(
                    im1, ax=ax[1],
                    location="bottom", shrink=1.0, label=f"Err: {title}")
                for iax in ax.flatten():
                    iax.grid(False)
                    iax.set_xlabel(self.xlabel)
                    iax.set_ylabel(self.ylabel)
                fig.tight_layout()
            if self.only_number:
                break
        return self
    
    def display_temperature(
        self,
        use_error_weightage: bool = True
    ) -> Self:
        if not use_error_weightage:
            h_widths_err = np.zeros_like(self.h_widths_err)
            v_widths_err = np.zeros_like(self.v_widths_err)
        else:
            h_widths_err = self.h_widths_err
            v_widths_err = self.v_widths_err
        self.horizontal_temperature = fit_linear(
            (self.unique_params1*1e-5)**2,
            self.h_widths_mean**2,
            2*self.h_widths_mean*h_widths_err
        )
        self.vertical_temperature = fit_linear(
            (self.unique_params1*1e-5)**2,
            self.v_widths_mean**2,
            2*self.v_widths_mean*v_widths_err
        )
        _h_temp = self.horizontal_temperature.slope*(self.mass*cn.u)/cn.k
        _v_temp = self.vertical_temperature.slope*(self.mass*cn.u)/cn.k
        _h_temp_err = \
            self.horizontal_temperature.slope_err*(self.mass*cn.u)/cn.k
        _v_temp_err = \
            self.vertical_temperature.slope_err*(self.mass*cn.u)/cn.k
        self.horizontal_temperature_value = _h_temp
        self.horizontal_temperature_error = _h_temp_err
        self.vertical_temperature_value = _v_temp
        self.vertical_temperature_error = _v_temp_err
        fig, ax = plt.subplots(1, 2, figsize=self.figsize)
        fig.subplots_adjust(wspace=0.02)
        _h_e = 2*self.h_widths_err*self.h_widths_mean
        _v_e = 2*self.v_widths_err*self.v_widths_mean
        ax[0].errorbar(
            1e6*self.horizontal_temperature.x,
            1e6*self.horizontal_temperature.y,
            yerr=np.abs(1e6*_h_e),
            fmt=self.fmt,
        )
        ax[0].plot(
            1e6*self.horizontal_temperature.x_fine,
            1e6*self.horizontal_temperature.y_fine,
            "-r"
        )
        ax[1].errorbar(
            1e6*self.vertical_temperature.x,
            1e6*self.vertical_temperature.y,
            yerr=np.abs(1e6*_v_e),
            fmt=self.fmt,
        )
        ax[1].plot(
            1e6*self.vertical_temperature.x_fine,
            1e6*self.vertical_temperature.y_fine,
            "-r"
        )
        ax[0].set_title(
            f"T$_v$: {1e6*_h_temp:.2f}$\pm${1e6*_h_temp_err:.2f} uK"
        )
        ax[1].set_title(
            f"T$_h$: {1e6*_v_temp:.2f}$\pm${1e6*_v_temp_err:.2f} uK"
        )
        ax[0].set_xlabel("Sq. expansion time [ms$^2$]")
        ax[1].set_xlabel("Sq. expansion time [ms$^2$]")
        ax[0].set_ylabel("Sq. vertical width [mm$^2$]")
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        ax[1].set_ylabel("Sq. horizontal width [mm$^2$]")
        return self
    
    def display_contour_images(
        self,
        *args,
        **kwargs
    ) -> Self:
        for fit in self.v_fits:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(fit.data)
            ax.contour(fit.data_fit, 8, colors='w')
            ax.grid(False)
        return self

    def display_images(
        self,
        *args,
        **kwargs
    ) -> Self:
        width_ax0 = 4.0
        width_ax1 = 1.0
        height_ax2 = 1.0
        left_margin = 0.65
        right_margin = 0.2
        bottom_margin = 0.5
        top_margin = 0.25
        inter_margin = 0.1
        factor = self.magnification/(self.binning*self.pixel_size)
        h, w = self.raw_images[0].shape
        height_ax0 = width_ax0 * float(h) / float(w)
        fwidth = left_margin + right_margin + \
            inter_margin + width_ax0 + width_ax1
        fheight = bottom_margin + top_margin + \
            inter_margin + height_ax0 + height_ax2
        n = len(self.raw_images)
        for k in range(n):
            raw_image = self.raw_images[k]
            processed_image = self.processed_images[k]
            _raw_height, _raw_width = raw_image.shape
            _raw_v_profile = np.zeros(_raw_height)
            _raw_h_profile = np.zeros(_raw_width)
            _raw_v_profile[self.col_start: self.col_end] = np.sum(
                processed_image,
                axis=1
            )
            _raw_h_profile[self.row_start: self.row_end] = np.sum(
                processed_image,
                axis=0
            )
            v_dim = np.arange(0, _raw_height)
            h_dim = np.arange(0, _raw_width)
            fig = plt.figure(figsize=(fwidth, fheight))
            ax0 = fig.add_axes(
                [
                    left_margin/fwidth,
                    (bottom_margin+inter_margin+height_ax2)/fheight,
                    width_ax0/fwidth,
                    height_ax0/fheight
                ]
            )
            ax0.xaxis.set_ticks_position("top")
            ax1 = fig.add_axes(
                [
                    (left_margin+width_ax0+inter_margin)/fwidth,
                    (bottom_margin+inter_margin+height_ax2) / fheight,
                    width_ax1/fwidth,
                    height_ax0/fheight
                ]
            )
            ax1.yaxis.tick_right()
            ax2 = fig.add_axes(
                [
                    left_margin/fwidth,
                    bottom_margin/fheight,
                    width_ax0/fwidth,
                    height_ax2/fheight
                ]
            )
            ax0.imshow(raw_image)
            ax0.grid(False)
            ax0.add_patch(
                Rectangle(
                    (self.row_start, self.col_start),
                    self.row_end-self.row_start,
                    self.col_end-self.col_start,
                    edgecolor='white',
                    facecolor='none',
                    fill=False,
                    lw=1
                )
            )
            ax1.plot(_raw_v_profile, v_dim, '.')
            ax1.set_ylim(_raw_height, 0)
            if self.h_fits[k] is not None:
                _h_fit: GaussianFitWithOffset = self.h_fits[k]
                ax1.plot(
                    _h_fit.y_fine,
                    self.col_start+_h_fit.x_fine*factor,
                    '-r'
                )
            ax2.plot(h_dim, _raw_h_profile, '.')
            ax2.set_xlim(0, _raw_width)
           
            if self.v_fits[k] is not None:
                _v_fit: GaussianFitWithOffset = self.v_fits[k]
                ax2.plot(
                    self.row_start+_v_fit.x_fine*factor,
                    _v_fit.y_fine,
                    '-r'
                )
            ax0.text(
                3, 3,
                f"file no. : {k+self.file_start}",
                fontsize=12,
                color="white",
                verticalalignment='top',
                bbox=dict(
                    boxstyle='round',
                    facecolor='lightsteelblue',
                    alpha=0.15
                )
            )
        return self
    
    def set_xscale(
        self,
        value: Union[int, float]
    ) -> Self:
        self.unique_params1 *= value
        return self

    def set_xoffset(
        self,
        value: Union[int, float]
    ) -> Self:
        self.unique_params1 -= value
        return self
    
    def set_yscale(
        self,
        value: Union[int, float]
    ) -> Self:
        self.yscale *= value
        return self

    def set_yoffset(
        self,
        value: Union[int, float]
    ) -> Self:
        self.yoffset -= value
        return self

    def set_xcalibration(
        self,
        values: List[Union[int, float]]
    ) -> Self:
        self.unique_params1 /= np.array(values)
        return self

    def set_xreplacement(
        self,
        values: List[Union[int, float, str]]
    ) -> Self:
        self.unique_params1 = np.array(values)
        return self
