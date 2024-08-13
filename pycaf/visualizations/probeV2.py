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
    GaussianFitWithoutOffset,
    ExponentialFitWithoutOffset
)
from ..analysis import (
    get_zip_archive,
    read_time_of_flight_from_zip,
    read_images_from_zip,
    read_parameters_from_zip,
    read_frequencies_from_zip,
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
        self.exposure_time_param = self.constants["cs_exposure_time_parameter"]
        self.gamma = self.constants["gamma"]
        self.collection_solid_angle = self.constants["collection_solid_angle"]
        self.magnification = self.constants["magnification"]
        self.pixel_size = self.constants["pixel_size"]
        self.binning = self.constants["binning"]
        self.mass = self.constants["mass"]
        self.year = year
        self.month = month
        self.day = day
        self.row_start = 0
        self.row_end = -1
        self.col_start = 0
        self.col_end = -1

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
            data_dict = _all_params | _all_frequencies
            assert parameter in data_dict
            params.append(data_dict[parameter])
        unique_params = list(set(params))
        unique_params.sort(reverse=np.sum(np.diff(params)) <= 0)
        unique_params = np.array(unique_params)
        return unique_params, data_dict

    def reset(self) -> Self:
        self.file_start: int = None
        self.file_stop: int = None
        self.unique_params: np.ndarray = None
        self.display_number_xlim: Tuple[float, float] = None
        self.display_horizontal_width_xlim: Tuple[float, float] = None
        self.display_vertical_width_xlim: Tuple[float, float] = None
        self.display_horizontal_centre_xlim: Tuple[float, float] = None
        self.display_vertical_centre_xlim: Tuple[float, float] = None
        self.display_number_ylim: Tuple[float, float] = None
        self.display_horizontal_width_ylim: Tuple[float, float] = None
        self.display_vertical_width_ylim: Tuple[float, float] = None
        self.display_horizontal_centre_ylim: Tuple[float, float] = None
        self.display_vertical_centre_ylim: Tuple[float, float] = None
        self.raw_images: np.ndarray = None
        self.processed_images: np.ndarray = None
        self.horizontal_fits: Dict[
            Tuple[int, int],
            GaussianFitWithoutOffset
        ] = {}
        self.vertical_fits: Dict[
            Tuple[int, int],
            GaussianFitWithoutOffset
        ] = {}
        self.tofs: np.ndarray = None
        self.tof_sampling_rate: int = None
        self.number: np.ndarray = None
        self.number_err: np.ndarray = 0.0
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
        return self

    def __call__(
        self,
        file_start: int,
        file_stop: int,
        parameter: Union[str, List[str]]
    ) -> Self:
        self.reset()
        self.file_start = file_start
        self.file_stop = file_stop
        if type(parameter) is str or \
                (type(parameter) is list and len(parameter) == 1):
            self.unique_params, self.data_dict = \
                self._get_unique_parameters(
                    file_start, file_stop, parameter
                )
            _trial_imgs = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    file_start, self.prefix
                )
            )
            _trial_img_dim = np.mean(_trial_imgs, axis=0).shape
            self.n_params = len(self.unique_params)
            self.n_iter = len(
                range(file_start, file_stop+1, 2*self.n_params)
            )
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
                    self.tofs[i, j, :] = yag_on_tof - yag_off_tof
                    _image = np.mean(yag_on - yag_off, axis=0)
                    self.raw_images[i, j, :, :] = _image
                    self.processed_images[i, j, :, :] = _image
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
            self.processed_images[:, :, row_start:row_end, col_start:col_end]
        return self

    def exclude_parameters_by_index(
        self,
        indices: List[int]
    ) -> Self:
        dims = self.processed_images.shape
        rev_indices = [i for i in range(dims[1]) if i not in indices]
        self.processed_images = \
            self.processed_images[:, rev_indices, :, :]
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
        _n: np.ndarray = number_multiplier*np.sum(
            self.processed_images,
            axis=(-1, -2)
        )
        self.number = _n.mean(axis=0)
        self.number_err = _n.std(axis=0)/np.sqrt(self.n_iter)
        return self

    def extract_shape(
        self,
        mean_images: bool = False
    ) -> Self:
        self.mean_images = mean_images
        if not mean_images:
            _h_width = np.zeros((self.n_iter, self.n_params))
            _v_width = np.zeros((self.n_iter, self.n_params))
            _h_centre = np.zeros((self.n_iter, self.n_params))
            _v_centre = np.zeros((self.n_iter, self.n_params))
            for j in range(self.n_params):
                for i in range(self.n_iter):
                    try:
                        v_fit, h_fit = \
                            calculate_cloud_size_from_image_1d_gaussian(
                                self.processed_images[i, j, :, :],
                                pixel_size=self.pixel_size,
                                bin_size=self.binning,
                                magnification=self.magnification
                            )
                        self.horizontal_fits[(i, j)] = h_fit
                        self.vertical_fits[(i, j)] = v_fit
                        _h_width[i, j] = h_fit.width
                        _v_width[i, j] = v_fit.width
                        _h_centre[i, j] = h_fit.centre
                        _v_centre[i, j] = v_fit.centre
                    except Exception as e:
                        file_no = self.file_start+j+i*self.n_iter
                        print(
                            f"Error {e} occured in file {file_no} in size fit"
                        )
            self.horizontal_width = _h_width.mean(axis=0)
            self.horizontal_centre = _h_centre.mean(axis=0)
            self.vertical_width = _v_width.mean(axis=0)
            self.vertical_centre = _v_centre.mean(axis=0)
            self.horizontal_width_err = \
                _h_width.std(axis=0)/np.sqrt(self.n_iter)
            self.horizontal_centre_err = \
                _h_centre.std(axis=0)/np.sqrt(self.n_iter)
            self.vertical_width_err = \
                _v_width.std(axis=0)/np.sqrt(self.n_iter)
            self.vertical_centre_err = \
                _v_centre.std(axis=0)/np.sqrt(self.n_iter)
        else:
            self.horizontal_width = np.zeros((self.n_params))
            self.horizontal_centre = np.zeros((self.n_params))
            self.vertical_width = np.zeros((self.n_params))
            self.vertical_centre = np.zeros((self.n_params))
            try:
                for j in range(self.n_params):
                    _processed_image = np.mean(
                        self.processed_images[:, j, :, :],
                        axis=0
                    )
                    v_fit, h_fit = \
                        calculate_cloud_size_from_image_1d_gaussian(
                            _processed_image,
                            pixel_size=self.pixel_size,
                            bin_size=self.binning,
                            magnification=self.magnification
                        )
                    self.horizontal_fits[(0, j)] = h_fit
                    self.vertical_fits[(0, j)] = v_fit
                    self.horizontal_width[j] = h_fit.width
                    self.vertical_width[j] = v_fit.width
                    self.horizontal_centre[j] = h_fit.centre
                    self.vertical_centre[j] = v_fit.centre
            except Exception as e:
                file_no = self.file_start+j*self.n_params  # FIXME
                print(
                    f"Error {e} occured in file {file_no} in size fit"
                )
        return self

    def extract_density(
        self
    ) -> Self:
        self.extract_number()
        self.extract_shape(mean_images=True)
        self.density = (3*self.number)/(
            4*np.pi*self.horizontal_width**2*self.vertical_width
        )
        self.density_err = np.abs(self.density)*np.sqrt(
            (self.number_err/self.number)**2 +
            (self.horizontal_width_err/self.horizontal_width)**2 +
            (self.vertical_width_err/self.vertical_width)**2
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
               self.horizontal_width,
               self.horizontal_width_err
            )
        if observable in vertical_width_aliases:
            self.vertical_width_fit: Fit = fitting_function(
                self.unique_params,
                self.vertical_width,
                self.vertical_width_err
            )
        if observable in horizontal_centre_aliases:
            self.horizontal_centre_fit: Fit = fitting_function(
                self.unique_params,
                self.horizontal_centre,
                self.horizontal_centre_err
            )
        if observable in vertical_centre_aliases:
            self.vertical_centre_fit: Fit = fitting_function(
                self.unique_params,
                self.vertical_centre,
                self.vertical_centre_err
            )
        return self

    def determine_temperature(
        self
    ) -> Self:
        self.extract_shape(mean_images=True)
        self.horizontal_temperature: LinearFit = fit_linear(
            self.unique_params**2,
            self.horizontal_width**2
        )
        self.vertical_temperature: LinearFit = fit_linear(
            self.unique_params**2,
            self.vertical_width**2
        )
        return self

    def determine_lifetime(
        self
    ) -> Self:
        self.extract_number()
        self.lifetime: ExponentialFitWithoutOffset = \
            fit_exponential_without_offset(
                self.unique_params,
                self.number,
                self.number_err
            )
        return self

    def plot_images(
        self,
        *args,
        **kwargs
    ) -> Self:
        width_ax0 = 4.
        width_ax1 = 1.
        height_ax2 = 1.
        left_margin = 0.65
        right_margin = 0.2
        bottom_margin = 0.5
        top_margin = 0.25
        inter_margin = 0.1

        _, _, h, w = self.raw_images.shape
        images = np.mean(self.raw_images, axis=0)
        height_ax0 = width_ax0 * float(h) / float(w)
        fwidth = left_margin + right_margin + \
            inter_margin + width_ax0 + width_ax1
        fheight = bottom_margin + top_margin + \
            inter_margin + height_ax0 + height_ax2

        for k in range(self.n_params):
            _image = images[k, :, :]
            h_profile = np.sum(_image, axis=0)
            h_dim = np.arange(0, len(h_profile))
            v_profile = np.sum(_image, axis=1)
            v_dim = np.arange(0, len(v_profile))
            fig = plt.figure(figsize=(fwidth, fheight))
            ax0 = fig.add_axes(
                [left_margin / fwidth,
                 (bottom_margin + inter_margin + height_ax2) / fheight,
                 width_ax0 / fwidth, height_ax0 / fheight]
            )
            ax1 = fig.add_axes(
                [(left_margin + width_ax0 + inter_margin) / fwidth,
                 (bottom_margin + inter_margin + height_ax2) / fheight,
                 width_ax1 / fwidth, height_ax0 / fheight]
            )
            ax2 = fig.add_axes(
                [left_margin / fwidth, bottom_margin / fheight,
                 width_ax0 / fwidth, height_ax2 / fheight]
            )
            ax0.imshow(images[k, :, :])
            ax0.add_patch(
                Rectangle(
                    (self.row_start, self.col_start),
                    self.row_end-self.row_start, self.col_end-self.col_start,
                    edgecolor='white',
                    facecolor='none',
                    fill=False,
                    lw=1
                )
            )
            ax0.xaxis.set_ticks_position("top")
            ax1.plot(v_profile, v_dim, '.')
            ax1.yaxis.tick_right()
            ax2.plot(h_dim, h_profile, '.')
            if len(self.horizontal_fits):
                if self.horizontal_fits[(0, k)] is not None:
                    _h_fit: GaussianFitWithoutOffset = \
                        self.horizontal_fits[(0, k)]
                    ax1.plot(_h_fit.x_fine, _h_fit.y_fine, '-r')
            if len(self.vertical_fits):
                if self.vertical_fits[(0, k)] is not None:
                    _v_fit: GaussianFitWithoutOffset = \
                        self.vertical_fits[(0, k)]
                    ax1.plot(_v_fit.x_fine, _v_fit.y_fine, '-r')
        return self

    def plot(
        self,
        **kwargs
    ) -> Self:
        list_of_y = [
            self.number,
            self.horizontal_width,
            self.vertical_width,
            self.horizontal_centre,
            self.vertical_centre,
            self.density
        ]
        list_of_yerr = [
            self.number_err,
            self.horizontal_width_err,
            self.vertical_width_err,
            self.horizontal_centre_err,
            self.vertical_centre_err,
            self.density_err
        ]
        list_of_ylabels = [
            "Molecule Numbers",
            "Horizontal Width [mm]",
            "Vertical Width [mm]",
            "Horizontal Centre [mm]",
            "Vertical Centre [mm]",
            "Density [Molecules/m3]"
        ]
        list_of_yfactors = [
            1.0,
            1e3,
            1e3,
            1e3,
            1e3,
            1.0
        ]
        list_of_yfits = [
            self.number_fit,
            self.horizontal_width_fit,
            self.vertical_width_fit,
            self.horizontal_centre_fit,
            self.vertical_centre_fit,
            self.density_fit
        ]
        for y, yerr, ylabel, yfactor, yfit in zip(
            list_of_y,
            list_of_yerr,
            list_of_ylabels,
            list_of_yfactors,
            list_of_yfits
        ):
            if y is not None:
                fig, ax = plt.subplots(1, 1, figsize=self.figsize)
                ax.errorbar(
                    self.unique_params,
                    y*yfactor,
                    yerr=yerr*yfactor,
                    fmt=self.fmt
                )
                ax.set_xlabel(self.xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(self.title)
                if yfit is not None:
                    ax.plot(
                        self.unique_params,
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
        if self.horizontal_temperature is not None:
            _h_temp = self.horizontal_temperature.slope*(self.mass*cn.u)/cn.k
            _v_temp = self.vertical_temperature.slope*(self.mass*cn.u)/cn.k
            fig, ax = plt.subplots(1, 2, figsize=self.figsize)
            ax[0].plot(
                1e6*self.horizontal_temperature.x,
                1e6*self.horizontal_temperature.y,
                self.fmt,
                1e6*self.horizontal_temperature.x_fine,
                1e6*self.horizontal_temperature.y_fine,
                "-r"
            )
            ax[1].plot(
                1e6*self.vertical_temperature.x,
                1e6*self.vertical_temperature.y,
                self.fmt,
                1e6*self.vertical_temperature.x_fine,
                1e6*self.vertical_temperature.y_fine,
                "-r"
            )
            ax[0].set_title(f"T_h: {1e6*_h_temp:.2f} uK")
            ax[1].set_title(f"T_v: {1e6*_v_temp:.2f} uK")
            ax[0].set_xlabel("Sq. expansion time [ms^2]")
            ax[1].set_xlabel("Sq. expansion time [ms^2]")
            ax[0].set_ylabel("Sq. horizontal width [mm^2]")
            ax[1].yaxis.set_label_position("right")
            ax[1].yaxis.tick_right()
            ax[1].set_ylabel("Sq. vertical width [mm^2]")
        if self.lifetime is not None:
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
        return self
