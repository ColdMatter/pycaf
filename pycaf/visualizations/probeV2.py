from typing import Tuple, List, Union, Dict, Callable
from typing_extensions import Self
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import json
import scipy.constants as cn
from scipy.ndimage import gaussian_filter

from ..analysis.models import (
    Fit,
    LinearFit,
    GaussianFitWithOffset,
    ExponentialFitWithoutOffset
)
from ..analysis import (
    get_zip_archive,
    read_time_of_flight_from_zip,
    read_images_from_zip,
    read_parameters_from_zip,
    read_frequencies_from_zip,
    read_ad9959_frequencies_from_zip,
    calculate_cloud_size_from_image_1d_gaussian,
    fit_linear,
    fit_quadratic_without_slope,
    fit_exponential_without_offset,
    fit_exponential_with_offset,
    fit_gaussian_with_offset,
    fit_gaussian_without_offset,
    fit_lorentzian_with_offset,
    fit_lorentzian_without_offset,
    fit_trap_frequency_oscillation
)

from .probeV1 import ProbeV1


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


# FIXME: Add multidimesional scan analysis
# FIXME: Add time of flight analysis and display
# FIXME: Add standerdized colorbar for all show_images
# FIXME: Filtering data by laser frequency/sideband discrepancy


class ProbeV2(ProbeV1):
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

    def _get_unique_parameters(
        self,
        file_start: int,
        file_stop: int,
        parameter: str
    ) -> Tuple[np.ndarray, Dict[str, Union[int, float]]]:
        params = []
        for fileno in range(file_start, file_stop+1, 2):
            _all_params = read_parameters_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            _all_frequencies = read_frequencies_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            _all_ad9959_frequencies = read_ad9959_frequencies_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            _data_dict = _all_params | _all_frequencies
            data_dict = _data_dict | _all_ad9959_frequencies
            assert parameter in data_dict
            params.append(data_dict[parameter])
        unique_params = list(set(params))
        unique_params.sort(reverse=np.sum(np.diff(params)) <= 0)
        unique_params = np.array(unique_params)
        return unique_params, data_dict

    def _get_unique_parameters_bgImage(
        self,
        file_start: int,
        file_stop: int,
        parameter: str
    ) -> Tuple[np.ndarray, Dict[str, Union[int, float]]]:
        params = []
        for fileno in range(file_start, file_stop+1, 1):
            _all_params = read_parameters_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            _all_frequencies = read_frequencies_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            _all_ad9959_frequencies = read_ad9959_frequencies_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            _data_dict = _all_params | _all_frequencies
            data_dict = _data_dict | _all_ad9959_frequencies
            assert parameter in data_dict
            if not data_dict[parameter] in params:
                params.append(data_dict[parameter])
            else:
                break
        unique_params = np.array(params)
        return unique_params, data_dict

    def reset(self) -> Self:
        self.file_start: int = None
        self.file_stop: int = None
        self.background_fileno: int = None
        self.unique_params: np.ndarray = None
        self.display_xlim: Tuple[float, float] = None
        self.display_number_ylim: Tuple[float, float] = None
        self.display_horizontal_width_ylim: Tuple[float, float] = None
        self.display_vertical_width_ylim: Tuple[float, float] = None
        self.display_horizontal_centre_ylim: Tuple[float, float] = None
        self.display_vertical_centre_ylim: Tuple[float, float] = None
        self.display_density_ylim: Tuple[float, float] = None
        self.raw_images: np.ndarray = None
        self.processed_images: np.ndarray = None
        self.processed_images_2: np.ndarray = None
        self.horizontal_fits: Dict[str, GaussianFitWithOffset] = {}
        self.vertical_fits: Dict[str, GaussianFitWithOffset] = {}
        self.tofs: np.ndarray = None
        self.tof_sampling_rate: int = None
        self.number: np.ndarray = None
        self.number_err: np.ndarray = 0.0
        self.number_full: np.ndarray = None
        self.number_full_err: np.ndarray = 0.0
        self.number_fit: Fit = None
        self.horizontal_width: np.ndarray = None
        self.horizontal_width_err: np.ndarray = 0.0
        self.horizontal_width_fit: Fit = None
        self.vertical_width: np.ndarray = None
        self.vertical_width_err: np.ndarray = 0.0
        self.vertical_width_fit: Fit = None
        self.horizontal_centre: np.ndarray = None
        self.horizontal_centre_err: np.ndarray = 0.0
        self.horizontal_centre_fit: Fit = None
        self.vertical_centre: np.ndarray = None
        self.vertical_centre_err: np.ndarray = 0.0
        self.vertical_centre_fit: Fit = None
        self.density: np.ndarray = None
        self.density_err: np.ndarray = 0.0
        self.density_fit: Fit = None
        self.horizontal_temperature: LinearFit = None
        self.vertical_temperature: LinearFit = None
        self.lifetime: ExponentialFitWithoutOffset = None
        self.mean_images: bool = False
        self.figsize: Tuple[int, int] = (8, 5)
        self.fmt: str = "ok"
        self.xlabel: str = None
        self.title: str = None
        self.row_start = 0
        self.row_end = -1
        self.col_start = 0
        self.col_end = -1
        return self

    def __call__(
        self,
        file_start: int,
        file_stop: int,
        parameter: Union[str, List[str]],
        discard_runs_upto: int = 0,
        is_bg_included: bool = False,
        is_norm_included: bool = False
    ) -> Self:
        self.reset()
        self.file_start = file_start
        self.file_stop = file_stop
        _trial_imgs = read_images_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month, self.day,
                file_start, self.prefix
            )
        )
        _trial_img_dim = np.mean(_trial_imgs, axis=0).shape
        if type(parameter) is str or \
                (type(parameter) is list and len(parameter) == 1):
            if not is_bg_included:
                self.unique_params, self.data_dict = \
                    self._get_unique_parameters(
                        file_start, file_stop, parameter
                    )
                self.n_params = len(self.unique_params)
                self.n_iter = len(
                    range(file_start, file_stop+1, 2*self.n_params)
                )
                self.n_sets = self.n_iter
                self.raw_images: np.ndarray = np.zeros(
                    (self.n_iter, self.n_params, *_trial_img_dim),
                    dtype=float
                )
                self.processed_images: np.ndarray = np.zeros(
                    (self.n_iter, self.n_params, *_trial_img_dim),
                    dtype=float
                )
                self.tofs: np.ndarray = np.zeros(
                    (self.n_iter, self.n_params, 1000),
                    dtype=float
                )
                for j in range(self.n_params):
                    for i, fileno in enumerate(
                        range(file_start+2*j, file_stop+2*j+1, 2*self.n_params)
                    ):
                        yag_on = read_images_from_zip(
                            get_zip_archive(
                                self.rootpath, self.year, self.month, self.day,
                                fileno, self.prefix
                            )
                        )
                        yag_off = read_images_from_zip(
                            get_zip_archive(
                                self.rootpath, self.year, self.month, self.day,
                                fileno+1, self.prefix
                            )
                        )
                        yag_on = yag_on[discard_runs_upto:]
                        yag_off = yag_off[discard_runs_upto:]
                        if len(yag_on) == len(yag_off):
                            _image = np.mean(yag_on-yag_off, axis=0)
                        else:
                            _image = np.mean(
                                yag_on-np.mean(yag_off, axis=0),
                                axis=0
                            )
                        self.raw_images[i, j, :, :] = _image
                        self.processed_images[i, j, :, :] = _image
                        self.tof_sampling_rate, yag_on_tof = \
                            read_time_of_flight_from_zip(
                                get_zip_archive(
                                    self.rootpath,
                                    self.year, self.month, self.day,
                                    fileno, self.prefix
                                )
                            )
                        _, yag_off_tof = read_time_of_flight_from_zip(
                            get_zip_archive(
                                self.rootpath, self.year, self.month, self.day,
                                fileno+1, self.prefix
                            )
                        )
                        if len(yag_on_tof) and len(yag_off_tof):
                            self.tofs[i, j, :] = yag_on_tof - yag_off_tof
            else:
                if not is_norm_included:
                    self.unique_params, self.data_dict = \
                        self._get_unique_parameters_bgImage(
                            file_start, file_stop, parameter
                        )
                    self.n_params = len(self.unique_params)
                    self.n_iter = len(
                        range(file_start, file_stop+1, self.n_params)
                    )
                    self.n_sets = self.n_iter
                    self.raw_images: np.ndarray = np.zeros(
                        (self.n_iter, self.n_params, *_trial_img_dim),
                        dtype=float
                    )
                    self.processed_images: np.ndarray = np.zeros(
                        (self.n_iter, self.n_params, *_trial_img_dim),
                        dtype=float
                    )
                    self.tofs: np.ndarray = np.zeros(
                        (self.n_iter, self.n_params, 1000),
                        dtype=float
                    )
                    for j in range(self.n_params):
                        for i, fileno in enumerate(
                            range(file_start+j, file_stop+j, self.n_params)
                        ):
                            _images = read_images_from_zip(
                                get_zip_archive(
                                    self.rootpath, self.year, self.month,
                                    self.day, fileno, self.prefix
                                )
                            )
                            # _images = _images[2*discard_runs_upto:]
                            _img = _images[0::2]
                            _bg = _images[1::2]
                            try:
                                _image = np.mean(_img - _bg, axis=0)
                                self.raw_images[i, j, :, :] = _image
                                self.processed_images[i, j, :, :] = _image
                            except Exception as e:
                                print(
                                    f"Error {e} occured in file" +
                                    f" {fileno} in Image"
                                )
                            self.tof_sampling_rate, yag_on_tof = \
                                read_time_of_flight_from_zip(
                                    get_zip_archive(
                                        self.rootpath,
                                        self.year, self.month, self.day,
                                        fileno, self.prefix
                                    )
                                )
                            if len(yag_on_tof):
                                self.tofs[i, j, :] = yag_on_tof
                else:
                    self.unique_params, self.data_dict = \
                        self._get_unique_parameters_bgImage(
                            file_start, file_stop, parameter
                        )
                    self.n_params = len(self.unique_params)
                    self.n_iter = len(
                        range(file_start, file_stop+1, self.n_params)
                    )
                    self.n_sets = self.n_iter
                    self.raw_images: np.ndarray = np.zeros(
                        (self.n_iter, self.n_params, *_trial_img_dim),
                        dtype=float
                    )
                    self.processed_images: np.ndarray = np.zeros(
                        (self.n_iter, self.n_params, *_trial_img_dim),
                        dtype=float
                    )
                    self.processed_images_2: np.ndarray = np.zeros(
                        (self.n_iter, self.n_params, *_trial_img_dim),
                        dtype=float
                    )
                    self.tofs: np.ndarray = np.zeros(
                        (self.n_iter, self.n_params, 1000),
                        dtype=float
                    )
                    for j in range(self.n_params):
                        for i, fileno in enumerate(
                            range(file_start+j, file_stop+j, self.n_params)
                        ):
                            _images = read_images_from_zip(
                                get_zip_archive(
                                    self.rootpath, self.year, self.month,
                                    self.day, fileno, self.prefix
                                )
                            )
                            # _images = _images[2*discard_runs_upto:]
                            _img_norm = _images[0::3]
                            _img = _images[1::3]
                            _bg = _images[2::3]
                            try:
                                _image_norm = np.mean(_img_norm - _bg, axis=0)
                                _image = np.mean(_img - _bg, axis=0)
                                _image_norm[_image_norm <= 0] = 1e-10
                                _image[_image <= 0] = 1e-10
                                self.raw_images[i, j, :, :] = \
                                    _image_norm
                                self.processed_images[i, j, :, :] = \
                                    _image
                                self.processed_images_2[i, j, :, :] = \
                                    _image_norm
                            except Exception as e:
                                print(
                                    f"Error {e} occured in file " +
                                    f"{fileno} in Image"
                                )
                            self.tof_sampling_rate, yag_on_tof = \
                                read_time_of_flight_from_zip(
                                    get_zip_archive(
                                        self.rootpath,
                                        self.year, self.month, self.day,
                                        fileno, self.prefix
                                    )
                                )
                            if len(yag_on_tof):
                                self.tofs[i, j, :] = yag_on_tof
        return self

    def set_roi(
        self,
        row_start: int = 0,
        row_end: int = -1,
        col_start: int = 0,
        col_end: int = -1,
    ) -> Self:
        self.row_start = row_start
        self.row_end = row_end
        self.col_start = col_start
        self.col_end = col_end
        self.processed_images = \
            self.processed_images[:, :, col_start:col_end, row_start:row_end]
        if self.processed_images_2:
            self.processed_images_2 = \
                self.processed_images_2[:, :, col_start:col_end, row_start:row_end]
        return self

    def exclude_parameters_by_index(
        self,
        indices: List[int]
    ) -> Self:
        dims = self.processed_images.shape
        rev_indices = [i for i in range(dims[1]) if i not in indices]
        self.processed_images = \
            self.processed_images[:, rev_indices, :, :]
        self.unique_params = self.unique_params[rev_indices]
        self.n_params -= len(indices)
        return self

    def exclude_iterations_by_index(
        self,
        indices: List[int]
    ) -> Self:
        dims = self.processed_images.shape
        rev_indices = [i for i in range(dims[0]) if i not in indices]
        self.processed_images = \
            self.processed_images[rev_indices, :, :, :]
        return self

    def extract_number(
        self
    ) -> Self:
        exposure_time = self.data_dict[self.exposure_time_param]*1e-5
        number_multiplier = self.photon/(
            exposure_time*self.gamma*self.collection_solid_angle
        )
        _n: np.ndarray = number_multiplier*np.nansum(
            self.processed_images,
            axis=(-1, -2)
        )
        self.number = np.nanmean(_n, axis=0)
        self.number_err = np.nanstd(_n, axis=0)/np.sqrt(self.n_iter)
        if self.processed_images_2:
            _n1: np.ndarray = number_multiplier*np.nansum(
                self.processed_images_2,
                axis=(-1, -2)
            )
            self.number_full = np.nanmean(_n/_n1, axis=0)
            self.number_full_err = np.nanstd(_n/_n1, axis=0)/np.sqrt(self.n_iter)
        return self

    def extract_shape(
        self,
        n_combine: int,
        combination_type: str = "sum",
        is_filter: bool = False,
        sigma_blur: float = 1.0
    ) -> Self:
        self.n_sets = int(self.n_iter/n_combine)
        self.horizontal_width = np.zeros((self.n_sets, self.n_params))
        self.horizontal_centre = np.zeros((self.n_sets, self.n_params))
        self.vertical_width = np.zeros((self.n_sets, self.n_params))
        self.vertical_centre = np.zeros((self.n_sets, self.n_params))
        for j in range(self.n_params):
            for i in range(self.n_sets):
                try:
                    _img = self.processed_images[
                            i*n_combine:(i+1)*n_combine, j, :, :
                        ].sum(axis=0)
                    if combination_type == "mean":
                        _img = self.processed_images[
                            i*n_combine:(i+1)*n_combine, j, :, :
                        ].mean(axis=0)
                    if is_filter:
                        _img = gaussian_filter(_img, sigma=sigma_blur)
                    v_fit, h_fit = \
                        calculate_cloud_size_from_image_1d_gaussian(
                            _img,
                            pixel_size=self.pixel_size,
                            bin_size=self.binning,
                            magnification=self.magnification
                        )
                    self.horizontal_fits[f"({i}, {j})"] = h_fit
                    self.vertical_fits[f"({i}, {j})"] = v_fit
                    if h_fit is not None:
                        self.horizontal_width[i, j] = h_fit.width
                        self.horizontal_centre[i, j] = h_fit.centre
                    if v_fit is not None:
                        self.vertical_width[i, j] = v_fit.width
                        self.vertical_centre[i, j] = v_fit.centre
                except Exception as e:
                    file_no = self.file_start+j+i*self.n_iter
                    print(
                        f"Error {e} occured in file {file_no} in size fit"
                    )
        return self

    def extract_density(
        self,
        n_combine: int,
        combination_type: str = "sum",
        is_filter: bool = False,
        sigma_blur: float = 1.0
    ) -> Self:
        self.extract_number()
        self.extract_shape(n_combine, combination_type, is_filter, sigma_blur)
        w_horiz = self.horizontal_width.mean(axis=0)
        w_horiz_err = self.horizontal_width.std(axis=0)/np.sqrt(self.n_sets)
        w_vert = self.vertical_width.mean(axis=0)
        w_vert_err = self.vertical_width.std(axis=0)/np.sqrt(self.n_sets)
        volume = (4*np.pi/3.0)*w_horiz**2*w_vert
        self.density = self.number/volume
        self.density_err = self.density*np.sqrt(
            (self.number_err/self.number)**2 +
            (2*w_horiz_err/w_horiz)**2 +
            (w_vert_err/w_vert)**2
        )
        return self

    def extract_temperature(
        self,
        n_combine: int,
        combination_type: str = "sum",
        is_filter: bool = False,
        sigma_blur: float = 1.0
    ) -> Self:
        self.extract_shape(n_combine, combination_type, is_filter, sigma_blur)
        self.horizontal_temperature = fit_linear(
            (self.unique_params*1e-5)**2,
            self.horizontal_width.mean(axis=0)**2,
            (self.horizontal_width.std(axis=0)/np.sqrt(self.n_sets))**2
        )
        self.vertical_temperature = fit_linear(
            (self.unique_params*1e-5)**2,
            self.vertical_width.mean(axis=0)**2,
            (self.vertical_width.std(axis=0)/np.sqrt(self.n_sets))**2
        )
        return self

    def curve_fit(
        self,
        observable: str,
        fitting_function: Callable
    ) -> Self:
        if observable in number_aliases:
            self.number_fit: Fit = fitting_function(
                self.unique_params,
                self.number,
                self.number_err
            )
        if observable in density_aliases:
            self.density_fit: Fit = fitting_function(
                self.unique_params,
                self.density,
                self.density_err
            )
        if observable in horizontal_width_aliases:
            self.horizontal_width_fit: Fit = fitting_function(
               self.unique_params,
               self.horizontal_width.mean(axis=0),
               self.horizontal_width.std(axis=0)/np.sqrt(self.n_sets)
            )
        if observable in vertical_width_aliases:
            self.vertical_width_fit: Fit = fitting_function(
                self.unique_params,
                self.vertical_width.mean(axis=0),
                self.vertical_width.std(axis=0)/np.sqrt(self.n_sets)
            )
        if observable in horizontal_centre_aliases:
            self.horizontal_centre_fit: Fit = fitting_function(
                self.unique_params,
                self.horizontal_centre.mean(axis=0),
                self.horizontal_centre.std(axis=0)/np.sqrt(self.n_sets)
            )
        if observable in vertical_centre_aliases:
            self.vertical_centre_fit: Fit = fitting_function(
                self.unique_params,
                self.vertical_centre.mean(axis=0),
                self.vertical_centre.std(axis=0)/np.sqrt(self.n_sets)
            )
        return self

    def show_images(
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
        _, _, h, w = self.raw_images.shape
        height_ax0 = width_ax0 * float(h) / float(w)
        fwidth = left_margin + right_margin + \
            inter_margin + width_ax0 + width_ax1
        fheight = bottom_margin + top_margin + \
            inter_margin + height_ax0 + height_ax2
        n_combine = int(self.n_iter/self.n_sets)
        for j in range(self.n_params):
            for i in range(self.n_sets):
                _processed_image = self.processed_images[
                    i*n_combine:(i+1)*n_combine, j, :, :
                ].sum(axis=0)
                _raw_image: np.ndarray = self.raw_images[
                    i*n_combine:(i+1)*n_combine, j, :, :
                ].sum(axis=0)
                _raw_height, _raw_width = _raw_image.shape
                _raw_v_profile = np.zeros(_raw_height)
                _raw_h_profile = np.zeros(_raw_width)
                _raw_v_profile[self.col_start: self.col_end] = np.sum(
                    _processed_image,
                    axis=1
                )
                _raw_h_profile[self.row_start: self.row_end] = np.sum(
                    _processed_image,
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
                ax0.imshow(_raw_image)
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
                if len(self.horizontal_fits):
                    if f"({i}, {j})" in self.horizontal_fits:
                        if self.horizontal_fits[f"({i}, {j})"] is not None:
                            _h_fit: GaussianFitWithOffset = \
                                self.horizontal_fits[f"({i}, {j})"]
                            ax1.plot(
                                _h_fit.y_fine,
                                self.col_start+_h_fit.x_fine*factor,
                                '-r'
                            )
                ax2.plot(h_dim, _raw_h_profile, '.')
                ax2.set_xlim(0, _raw_width)
                if len(self.vertical_fits):
                    if f"({i}, {j})" in self.vertical_fits:
                        if self.vertical_fits[f"({i}, {j})"] is not None:
                            _v_fit: GaussianFitWithOffset = \
                                self.vertical_fits[f"({i}, {j})"]
                            ax2.plot(
                                self.row_start+_v_fit.x_fine*factor,
                                _v_fit.y_fine,
                                '-r'
                            )
                ax0.text(
                    3, 3,
                    f"parameter: {j}, set: {i}",
                    fontsize=8,
                    color="white",
                    verticalalignment='top',
                    bbox=dict(
                        boxstyle='round',
                        facecolor='lightsteelblue',
                        alpha=0.15
                    )
                )
        return self

    def plot(
        self,
        **kwargs
    ) -> Self:
        _h_width_mean, _v_width_mean = None, None
        _h_width_err, _v_width_err = None, None
        _h_centre_mean, _v_centre_mean = None, None
        _h_centre_err, _v_centre_err = None, None
        if self.horizontal_width is not None:
            _h_width_mean = self.horizontal_width.mean(axis=0)
            _h_width_err = self.horizontal_width.std(axis=0) /\
                np.sqrt(self.n_sets)
        if self.vertical_width is not None:
            _v_width_mean = self.vertical_width.mean(axis=0)
            _v_width_err = self.vertical_width.std(axis=0) /\
                np.sqrt(self.n_sets)
        if self.horizontal_centre is not None:
            _h_centre_mean = self.horizontal_centre.mean(axis=0)
            _h_centre_err = self.horizontal_centre.std(axis=0) /\
                np.sqrt(self.n_sets)
        if self.vertical_width is not None:
            _v_centre_mean = self.vertical_centre.mean(axis=0)
            _v_centre_err = self.vertical_centre.std(axis=0) /\
                np.sqrt(self.n_sets)
        list_of_y: List[np.ndarray] = [
            self.number,
            _h_width_mean,
            _v_width_mean,
            _h_centre_mean,
            _v_centre_mean,
            self.density
        ]
        list_of_yerr: List[np.ndarray] = [
            self.number_err,
            _h_width_err,
            _v_width_err,
            _h_centre_err,
            _v_centre_err,
            self.density_err
        ]
        list_of_ylabels: List[str] = [
            "Molecule Numbers",
            "Vertical Width [mm]",
            "Horizontal Width [mm]",
            "Vertical Centre [mm]",
            "Horizontal Centre [mm]",
            "Density [Molecules/m3]"
        ]
        list_of_yfactors: List[float] = [
            1.0,
            1e3,
            1e3,
            1e3,
            1e3,
            1.0
        ]
        list_of_yfits: List[Fit] = [
            self.number_fit,
            self.horizontal_width_fit,
            self.vertical_width_fit,
            self.horizontal_centre_fit,
            self.vertical_centre_fit,
            self.density_fit
        ]
        list_of_ylimits: List[Tuple[float, float]] = [
            self.display_number_ylim,
            self.display_horizontal_width_ylim,
            self.display_vertical_width_ylim,
            self.display_horizontal_centre_ylim,
            self.display_vertical_centre_ylim,
            self.display_density_ylim
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
                    self.unique_params,
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
        if self.horizontal_temperature is not None \
                and self.vertical_temperature is not None:
            self._plot_temperature()
        if self.lifetime is not None:
            self._plot_lifetime()
        return self

    def _plot_temperature(
        self
    ) -> Self:
        _h_temp = self.horizontal_temperature.slope*(self.mass*cn.u)/cn.k
        _v_temp = self.vertical_temperature.slope*(self.mass*cn.u)/cn.k
        _h_temp_err = \
            self.horizontal_temperature.slope_err*(self.mass*cn.u)/cn.k
        _v_temp_err = \
            self.vertical_temperature.slope_err*(self.mass*cn.u)/cn.k
        fig, ax = plt.subplots(1, 2, figsize=self.figsize)
        fig.subplots_adjust(wspace=0.02)
        _h_w_e = self.horizontal_width.std(axis=0)/np.sqrt(self.n_sets)
        _v_w_e = self.vertical_width.std(axis=0)/np.sqrt(self.n_sets)
        _h_e = 2*self.horizontal_width.mean(axis=0)*_h_w_e
        _v_e = 2*self.vertical_width.mean(axis=0)*_v_w_e
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
            f"T$_v$: {1e6*_h_temp:.2f}+/-{1e6*_h_temp_err:.2f} uK"
        )
        ax[1].set_title(
            f"T$_h$: {1e6*_v_temp:.2f}+/-{1e6*_v_temp_err:.2f} uK"
        )
        ax[0].set_xlabel("Sq. expansion time [ms$^2$]")
        ax[1].set_xlabel("Sq. expansion time [ms$^2$]")
        ax[0].set_ylabel("Sq. vertical width [mm$^2$]")
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        ax[1].set_ylabel("Sq. horizontal width [mm$^2$]")
        return self

    def _plot_lifetime(
        self
    ) -> Self:
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.plot(
            self.lifetime.x_fine,
            self.lifetime.y_fine,
            "-r"
        )
        ax.set_title(f"Trap lifetime: {self.lifetime.rate*1e3:.2f} ms")
        return self

    def set_figsize(
        self,
        value: Tuple[int, int]
    ) -> Self:
        self.figsize = value
        return self

    def set_title(
        self,
        value: str
    ) -> Self:
        self.title = value
        return self

    def set_plot_fmt(
        self,
        value: str
    ) -> Self:
        self.fmt = value
        return self

    def set_xscale(
        self,
        value: Union[int, float]
    ) -> Self:
        self.unique_params *= value
        return self

    def set_xoffset(
        self,
        value: Union[int, float]
    ) -> Self:
        self.unique_params -= value
        return self

    def set_xcalibration(
        self,
        values: List[Union[int, float]]
    ) -> Self:
        self.unique_params /= np.array(values)
        return self

    def set_xreplacement(
        self,
        values: List[Union[int, float, str]]
    ) -> Self:
        self.unique_params = np.array(values)
        return self

    def set_xlim(
        self,
        value: Tuple[float, float]
    ) -> Self:
        self.display_xlim = value
        return self

    def set_xlabel(
        self,
        value: str
    ) -> Self:
        self.xlabel = value
        return self
    
    def set_ylabel(
        self,
        value: str
    ) -> Self:
        self.ylabel = value
        return self

    def set_ylim(
        self,
        observable: str,
        value: Tuple[float, float]
    ) -> Self:
        if observable in number_aliases:
            self.display_number_ylim = value
        if observable in horizontal_width_aliases:
            self.display_horizontal_width_ylim = value
        if observable in vertical_width_aliases:
            self.display_vertical_width_ylim = value
        if observable in horizontal_centre_aliases:
            self.display_horizontal_centre_ylim = value
        if observable in vertical_centre_aliases:
            self.display_vertical_centre_ylim = value
        if observable in density_aliases:
            self.display_density_ylim = value
        return self

    def single_shot_mot_lifetime(
        self,
        fileno,
        row_start: int = 0,
        row_end: int = -1,
        col_start: int = 0,
        col_end: int = -1,
        n_iter: int = 0
    ) -> Self:
        _images = read_images_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month,
                self.day, fileno, self.prefix
            )
        )
        _params = read_parameters_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month,
                self.day, fileno, self.prefix
            )
        )
        n_images = int(_params["NumberOfMOTImages"])+1
        if n_iter == 0:
            n_iter = int(len(_images)/n_images)
        t_sep = _params["ConsequtiveImageSeparation"]
        start = _params["ImageAquireStart"]
        t_array = 1e-2*np.array([start+i*t_sep for i in range(n_images-1)])
        fractions = []
        first_image = []
        for i in range(1, n_iter):
            one_mot_iter = _images[i*n_images:(i+1)*n_images, :, :]
            one_mot_iter -= one_mot_iter[-1, :, :]
            one_mot_iter = one_mot_iter[:,
                                        row_start:row_end,
                                        col_start:col_end]
            first_image.append(one_mot_iter[0, :, :])
            n = np.sum(one_mot_iter, axis=(1, 2))
            n /= n[0]
            n = n[:-1]
            fractions.append(n)
        fractions = np.array(fractions).reshape((n_iter-1, n_images-1))
        fractions_mean = fractions.mean(axis=0)
        fractions_err = fractions.std(axis=0)/np.sqrt(n_iter)
        fit = fit_exponential_without_offset(
            t_array,
            fractions_mean
        )
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.errorbar(t_array, fractions_mean, yerr=fractions_err, fmt="ok")
        ax.plot(fit.x_fine, fit.y_fine, "-r")
        ax.set_xlabel("CMOT Time [ms]")
        ax.set_ylabel("MOT retaining fraction")
        ax.set_title(f"MOT lifetime: {fit.rate:.2f} ms")
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(np.array(first_image).mean(axis=0))
        ax.grid(False)
        return self
