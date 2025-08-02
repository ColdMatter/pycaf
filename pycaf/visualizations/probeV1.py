from typing import Tuple, List, Union, Dict, Callable
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import json
import scipy.constants as cn

from ..analysis.models import Fit
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


class ProbeV1():
    def __init__(
        self,
        config_path: str,
        year: int,
        month: int,
        day: int,
    ) -> None:
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
        self.year = year
        self.month = month
        self.day = day

    def _1D_plot(
        self,
        file_start: int,
        file_stop: int,
        x: np.ndarray,
        y_mean: np.ndarray,
        y_err: np.ndarray = None,
        fit: Fit = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        if "figsize" in kwargs:
            figsize = kwargs["figsize"]
        else:
            figsize = (8, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if "fmt" in kwargs:
            fmt = kwargs["fmt"]
        else:
            fmt = "ok"
        if "label" in kwargs:
            label = kwargs["label"]
        else:
            label = None
        if y_err is not None:
            ax.errorbar(
                x,
                y_mean,
                yerr=y_err,
                fmt=fmt,
                label=label
            )
        else:
            ax.plot(x, y_mean, fmt, label=label)
        date_str = f"{self.year}/{self.month}/{self.day}"
        file_str = f"{file_start}-{file_stop}"
        ax.text(
            1.03, 0.98,
            f"File Info: {date_str}: {file_str}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='lightsteelblue',
                alpha=0.15
            )
        )
        if fit:
            ax.plot(fit.x_fine, fit.y_fine, "-r", label="Fit")
            ax.text(
                1.03, 0.9,
                "Fitting info:"+fit.func_str+fit.args_str,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(
                    boxstyle='round',
                    facecolor='lightsteelblue',
                    alpha=0.15
                )
            )
        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"])
        if "title" in kwargs:
            ax.set_title(kwargs["title"])
        if "xlim" in kwargs:
            ax.set_xlim(kwargs["xlim"])
        if "ylim" in kwargs:
            ax.set_ylim(kwargs["ylim"])
        if label is not None:
            ax.legend()
        return fig, ax

    def _1D_fit(
        self,
        fitting: str,
        x: np.ndarray,
        y_mean: np.ndarray,
        y_err: np.ndarray = None,
    ) -> Fit:
        fit = None
        if fitting in fitting_func_map:
            fit = fitting_func_map[fitting](
                    x,
                    y_mean,
                    y_err
                )
        return fit

    def _plot_cloud_images(
        self,
        parameter: str,
        unique_params: np.ndarray,
        mean_cropped_images: np.ndarray,
        n: np.ndarray,
        dn: np.ndarray,
        v_fits: Dict[int, Fit],
        h_fits: Dict[int, Fit],
        row_start,
        row_end,
        col_start,
        col_end,
        **kwargs
    ) -> None:
        if "cloud_image_figsize" in kwargs:
            figsize = kwargs["cloud_image_figsize"]
        else:
            figsize = (12, 4)
        for j in range(len(unique_params)):
            fig = plt.figure(figsize=figsize)
            fig.suptitle(f"{parameter}: {unique_params[j]}")
            ax0 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
            ax1 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
            _im = ax0.imshow(mean_cropped_images[j, :, :])
            if "clim" in kwargs:
                _im.set_clim(kwargs["clim"])
            if "grid" in kwargs:
                ax0.grid(kwargs["grid"])
            if "show_roi" in kwargs:
                if kwargs["show_roi"]:
                    ax0.add_patch(
                        Rectangle(
                            (row_start, col_start),
                            row_end-row_start, col_end-col_start,
                            edgecolor='white',
                            facecolor='none',
                            fill=False,
                            lw=1
                        )
                    )
            ax0.text(5, 10, f"N: {n[j]:.0f}+/-{dn[j]:.0f}", color="w")
            if j in h_fits:
                ax1.plot(
                    1e3*h_fits[j].x, h_fits[j].y, ".g", label="raw data"
                )
                ax1.plot(
                    1e3*h_fits[j].x_fine,
                    h_fits[j].y_fine, "-g",
                    label="horizntal fit"
                )
            if j in v_fits:
                ax1.plot(
                    1e3*v_fits[j].x,
                    v_fits[j].y, ".b",
                    label="raw data"
                )
                ax1.plot(
                    1e3*v_fits[j].x_fine,
                    v_fits[j].y_fine, "-b",
                    label="vertical fit"
                )
            ax1.set_xlabel("Distance [mm]")
            ax1.set_ylabel("Integrated Cluod Signal [a. u.]")
            ax1.yaxis.set_label_position("right")
            ax1.yaxis.tick_right()
            ax1.legend()
        return None

    def get_unique_parameters(
        self,
        file_start: int,
        file_stop: int,
        parameter: str
    ) -> Tuple[List[Union[int, float]], Dict[str, Union[int, float]]]:
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
        return unique_params, data_dict
    
    def get_unique_parameters_bgImage(
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
        unique_params = np.array(params)
        return unique_params, data_dict

    def single_parameter_cloud_characterization(
        self,
        file_start: int,
        file_stop: int,
        parameter: str,
        row_start: int = 0,
        row_end: int = -1,
        col_start: int = 0,
        col_end: int = -1,
        filter_points_for_fit: List[int] = [],
        filter_iterations: List[int] = [],
        parameter_callback: Callable = None,
        number_callback: Callable = None,
        width_callback: Callable = None,
        centre_callback: Callable = None,
        fit_number: Callable = None,
        fit_horizontal_width: Callable = None,
        fit_vertical_width: Callable = None,
        fit_horizontal_centre: Callable = None,
        fit_vertical_centre: Callable = None,
        fit_density: Callable = None,
        display_cloud_images: bool = False,
        **kwargs
    ) -> None:
        xscale, xoffset, yscale, yoffset = 1.0, 0.0, 1.0, 0.0
        n_fit, h_width_fit, h_centre_fit = None, None, None
        density_fit, v_width_fit, v_centre_fit = None, None, None
        n_th = 100
        unique_params, data_dict = self.get_unique_parameters(
            file_start, file_stop, parameter
        )
        mean_imgs, v_fits, h_fits = [], {}, {}
        n = np.zeros(len(unique_params))
        dn = np.zeros(len(unique_params))
        h_width = np.zeros(len(unique_params))
        v_width = np.zeros(len(unique_params))
        h_centre = np.zeros(len(unique_params))
        v_centre = np.zeros(len(unique_params))
        density = np.zeros(len(unique_params))
        exposure_time = data_dict[self.exposure_time_param]*1e-5
        number_multiplier = self.photon/(
            exposure_time*self.constants["gamma"]
            * self.constants["collection_solid_angle"]
        )
        if "number_threshold" in kwargs:
            n_th = kwargs["number_threshold"]
        for j in range(len(unique_params)):
            _img_array: List[np.ndarray] = []
            for i, fileno in enumerate(
                range(file_start+2*j, file_stop+2*j+1, 2*len(unique_params))
            ):
                if i not in filter_iterations:
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
                    _img_array.append(np.mean(yag_on - yag_off, axis=0))
            img_array: np.ndarray = np.array(_img_array)
            cropped_img_array: np.ndarray = \
                img_array[:, row_start:row_end, col_start:col_end]
            sum_cropped_img_array: np.ndarray = \
                cropped_img_array.sum(axis=0)
            mean_img_array: np.ndarray = img_array.mean(axis=0)
            mean_imgs.append(mean_img_array)
            _n: np.ndarray = number_multiplier*np.sum(
                img_array[:, col_start:col_end, row_start:row_end],
                axis=(1, 2)
            )
            n[j] = _n.mean()
            dn[j] = _n.std()/np.sqrt(i+1)
            if n[j] > n_th:
                v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                    sum_cropped_img_array,
                    pixel_size=self.constants["pixel_size"],
                    bin_size=self.constants["binning"],
                    magnification=self.constants["magnification"]
                )
                if v_fit is not None and h_fit is not None:
                    v_fits[j] = v_fit
                    h_fits[j] = h_fit
                    v_centre[j] = v_fit.centre
                    h_centre[j] = h_fit.centre
                    v_width[j] = v_fit.width
                    h_width[j] = h_fit.width
                    volume = (4.0/3.0)*np.pi*v_width[j]*h_width[j]**2
                    density[j] = n[j]/volume

        if display_cloud_images:
            self._plot_cloud_images(
                parameter, unique_params,
                np.array(mean_imgs),
                n, dn, v_fits, h_fits,
                row_start, row_end,
                col_start, col_end,
                **kwargs
            )

        if parameter_callback is not None:
            unique_params = parameter_callback(unique_params, **kwargs)
        if number_callback is not None:
            n = number_callback(n, **kwargs)
            dn = number_callback(dn, **kwargs)
        if width_callback is not None:
            v_width = width_callback(v_width, **kwargs)
            h_width = width_callback(h_width, **kwargs)
        if centre_callback is not None:
            v_centre = centre_callback(v_centre, **kwargs)
            h_centre = centre_callback(h_centre, **kwargs)

        if "xscale" in kwargs:
            xscale = kwargs["xscale"]
        if "xoffset" in kwargs:
            xoffset = kwargs["xoffset"]
        if "yscale" in kwargs:
            yscale = kwargs["yscale"]
        if "yoffset" in kwargs:
            yoffset = kwargs["yoffset"]

        unique_params = xscale*(np.array(unique_params)-xoffset)
        v_centre = yscale*(v_centre-yoffset)
        h_centre = yscale*(h_centre-yoffset)
        v_width = yscale*(v_width-yoffset)
        h_width = yscale*(h_width-yoffset)

        filtered_unique_params = np.delete(
            unique_params,
            filter_points_for_fit
        )

        if fit_number is not None:
            n_fit = fit_number(
                filtered_unique_params,
                np.delete(n, filter_points_for_fit),
                np.delete(dn, filter_points_for_fit)
            )
        if fit_horizontal_width is not None:
            h_width_fit = fit_horizontal_width(
                filtered_unique_params,
                np.delete(h_width, filter_points_for_fit)
            )
        if fit_vertical_width is not None:
            v_width_fit = fit_vertical_width(
                filtered_unique_params,
                np.delete(v_width, filter_points_for_fit)
            )
        if fit_horizontal_centre is not None:
            h_centre_fit = fit_horizontal_centre(
                filtered_unique_params,
                np.delete(h_centre, filter_points_for_fit)
            )
        if fit_vertical_centre is not None:
            v_centre_fit = fit_vertical_centre(
                filtered_unique_params,
                np.delete(v_centre, filter_points_for_fit)
            )
        if fit_density in kwargs:
            density_fit = fit_density(
                filtered_unique_params,
                np.delete(density, filter_points_for_fit)
            )

        if "display_number_variation" in kwargs:
            self._1D_plot(
                file_start, file_stop,
                unique_params, y_mean=n, y_err=dn,
                fit=n_fit,
                ylabel="Molecule number",
                **kwargs
            )
        if "display_size_variation" in kwargs:
            if kwargs["display_size_variation"]:
                self._1D_plot(
                    file_start, file_stop,
                    unique_params, y_mean=h_width, y_err=None,
                    fit=h_width_fit,
                    ylabel="Horizontal width",
                    **kwargs
                )
                self._1D_plot(
                    file_start, file_stop,
                    unique_params, y_mean=v_width, y_err=None,
                    fit=v_width_fit,
                    ylabel="Vertical width",
                    **kwargs
                )
        if "display_position_variation" in kwargs:
            if kwargs["display_position_variation"]:
                self._1D_plot(
                    file_start, file_stop,
                    unique_params, y_mean=h_centre, y_err=None,
                    fit=h_centre_fit,
                    ylabel="Horizontal centre",
                    **kwargs
                )
                self._1D_plot(
                    file_start, file_stop,
                    unique_params, y_mean=v_centre, y_err=None,
                    fit=v_centre_fit,
                    ylabel="Vertical centre",
                    **kwargs
                )
        if "display_density_variation" in kwargs:
            self._1D_plot(
                file_start, file_stop,
                unique_params, y_mean=density, y_err=None,
                fit=density_fit,
                ylabel="Density",
                **kwargs
            )
        return None

    def number_by_image(
        self,
        file_start: int,
        file_stop: int,
        parameter: str,
        row_start: int = 0,
        row_end: int = -1,
        col_start: int = 0,
        col_end: int = -1,
        fitting: str = None,
        param_index_fit_exclude: List[int] = [],
        params_calibration: List[float] = [],
        **kwargs
    ) -> Tuple[Fit, np.ndarray]:
        unique_params, data_dict = self.get_unique_parameters(
            file_start, file_stop, parameter
        )
        _imgs = read_images_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month, self.day,
                file_start, self.prefix
            )
        )
        imgs = np.zeros(
            (len(unique_params), _imgs.shape[-2], _imgs.shape[-1]),
            dtype=float
        )
        n = np.zeros(len(unique_params))
        dn = np.zeros(len(unique_params))
        exposure_time = data_dict[self.exposure_time_param]*1e-5
        number_multiplier = self.photon/(
            exposure_time*self.constants["gamma"]
            * self.constants["collection_solid_angle"]
        )
        for j in range(len(unique_params)):
            _img_array: List[np.ndarray] = []
            for i, fileno in enumerate(
                range(file_start+2*j, file_stop+2*j+1, 2*len(unique_params))
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
                _img_array.append(np.mean(yag_on - yag_off, axis=0))
            img_array: np.ndarray = np.array(_img_array)
            imgs[j, :, :] = img_array.mean(axis=0)
            _n: np.ndarray = number_multiplier*np.sum(
                img_array[:, col_start:col_end, row_start:row_end],
                axis=(1, 2)
            )
            n[j] = _n.mean()
            dn[j] = _n.std()/np.sqrt(i+1)

        if "xscale" in kwargs:
            xscale = kwargs["xscale"]
        else:
            xscale = 1.0
        if "xoffset" in kwargs:
            xoffset = kwargs["xoffset"]
        else:
            xoffset = 0.0

        params = xscale*(np.array(unique_params)-xoffset)
        if len(params_calibration):
            params /= np.array(params_calibration)
        params_excluded = np.delete(params, param_index_fit_exclude)
        n_excluded = np.delete(n, param_index_fit_exclude)
        dn_excluded = np.delete(dn, param_index_fit_exclude)

        fit = self._1D_fit(
            fitting,
            params_excluded,
            n_excluded,
            dn_excluded
        )
        _, ax = self._1D_plot(
            file_start, file_stop,
            params, y_mean=n, y_err=dn,
            fit=fit,
            **kwargs
        )
        if "display_images" in kwargs:
            if kwargs["display_images"]:
                if "image_figsize" in kwargs:
                    figsize = kwargs["image_figsize"]
                else:
                    figsize = (4, 4)
                for j in range(len(unique_params)):
                    _, ax = plt.subplots(1, 1, figsize=figsize)
                    _im = ax.imshow(imgs[j, :, :])
                    ax.set_title(f"{parameter}: {unique_params[j]}")
                    if "clim" in kwargs:
                        _im.set_clim(kwargs["clim"])
                    if "grid" in kwargs:
                        ax.grid(kwargs["grid"])
                    if "show_roi" in kwargs:
                        if kwargs["show_roi"]:
                            ax.add_patch(
                                Rectangle(
                                    (row_start, col_start),
                                    row_end-row_start, col_end-col_start,
                                    edgecolor='white',
                                    facecolor='none',
                                    fill=False,
                                    lw=1
                                )
                            )
        return fit, imgs

    def size_by_image(
        self,
        file_start: int,
        file_stop: int,
        parameter: str,
        row_start: int = 0,
        row_end: int = -1,
        col_start: int = 0,
        col_end: int = -1,
        fitting: str = None,
        param_index_fit_exclude: List[int] = [],
        **kwargs
    ) -> Tuple[Fit, Fit, np.ndarray]:
        unique_params, data_dict = self.get_unique_parameters(
            file_start, file_stop, parameter
        )
        _imgs = read_images_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month, self.day,
                file_start, self.prefix
            )
        )
        imgs = np.zeros(
            (len(unique_params), _imgs.shape[-2], _imgs.shape[-1]),
            dtype=float
        )
        h_width = np.zeros(len(unique_params))
        v_width = np.zeros(len(unique_params))
        for j in range(len(unique_params)):
            _img_array: List[np.ndarray] = []
            for i, fileno in enumerate(
                range(file_start+2*j, file_stop+2*j+1, 2*len(unique_params))
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
                _img_array.append(np.mean(yag_on - yag_off, axis=0))
            img_array: np.ndarray = np.array(_img_array)
            imgs[j, :, :] = img_array.mean(axis=0)
            v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                imgs[j, col_start:col_end, row_start:row_end],
                pixel_size=self.constants["pixel_size"],
                bin_size=self.constants["binning"],
                magnification=self.constants["magnification"]
            )
            v_width[j] = v_fit.width
            h_width[j] = h_fit.width
            if "display_fits" in kwargs:
                if kwargs["display_fits"]:
                    _, ax = plt.subplots(1, 1, figsize=(8, 5))
                    ax.plot(
                        1e3*v_fit.x, v_fit.y, "ok",
                        1e3*v_fit.x_fine, v_fit.y_fine, "-r"
                    )
                    ax.set_xlabel("Vertical distance [mm]")
                    ax.set_ylabel("Integrated profie")
                    _, ax = plt.subplots(1, 1, figsize=(8, 5))
                    ax.plot(
                        1e3*h_fit.x, h_fit.y, "ok",
                        1e3*h_fit.x_fine, h_fit.y_fine, "-r"
                    )
                    ax.set_xlabel("Horizntal distance [mm]")
                    ax.set_ylabel("Integrated profie")

        if "xscale" in kwargs:
            xscale = kwargs["xscale"]
        else:
            xscale = 1.0
        if "xoffset" in kwargs:
            xoffset = kwargs["xoffset"]
        else:
            xoffset = 0.0
        if "yscale" in kwargs:
            yscale = kwargs["yscale"]
        else:
            yscale = 1.0
        if "yoffset" in kwargs:
            yoffset = kwargs["yoffset"]
        else:
            yoffset = 0.0

        params = xscale*(np.array(unique_params)-xoffset)
        v_width = yscale*(v_width-yoffset)
        h_width = yscale*(h_width-yoffset)
        params_excluded = np.delete(params, param_index_fit_exclude)
        h_width_excluded = np.delete(h_width, param_index_fit_exclude)
        v_width_excluded = np.delete(v_width, param_index_fit_exclude)

        h_width_fit = self._1D_fit(
            fitting,
            params_excluded,
            h_width_excluded
        )
        v_width_fit = self._1D_fit(
            fitting,
            params_excluded,
            v_width_excluded
        )

        _, ax = self._1D_plot(
            file_start, file_stop,
            params, y_mean=h_width, y_err=None,
            fit=h_width_fit,
            title="Horizontal width",
            **kwargs
        )
        _, ax = self._1D_plot(
            file_start, file_stop,
            params, y_mean=v_width, y_err=None,
            fit=v_width_fit,
            title="Vertical width",
            **kwargs
        )
        if "display_images" in kwargs:
            if kwargs["display_images"]:
                if "image_figsize" in kwargs:
                    figsize = kwargs["image_figsize"]
                else:
                    figsize = (4, 4)
                for j in range(len(unique_params)):
                    _, ax = plt.subplots(1, 1, figsize=figsize)
                    _im = ax.imshow(imgs[j, :, :])
                    ax.set_title(f"{parameter}: {unique_params[j]}")
                    if "clim" in kwargs:
                        _im.set_clim(kwargs["clim"])
                    if "grid" in kwargs:
                        ax.grid(kwargs["grid"])
                    if "show_roi" in kwargs:
                        if kwargs["show_roi"]:
                            ax.add_patch(
                                Rectangle(
                                    (row_start, col_start),
                                    row_end-row_start, col_end-col_start,
                                    edgecolor='white',
                                    facecolor='none',
                                    fill=False,
                                    lw=1
                                )
                            )
        return v_width_fit, h_width_fit, imgs

    def number_by_tof(
        self,
        file_start: int,
        file_stop: int,
        parameter: str,
        bin_start: int = 0,
        bin_end: int = -1,
        fitting: str = None,
        param_index_fit_exclude: List[int] = [],
        **kwargs
    ) -> Tuple[Fit, np.ndarray]:
        unique_params, data_dict = self.get_unique_parameters(
            file_start, file_stop, parameter
        )
        tofs = np.zeros((len(unique_params), 1000), dtype=float)
        n = np.zeros(len(unique_params))
        dn = np.zeros(len(unique_params))
        for j in range(len(unique_params)):
            _tof_array: List[np.ndarray] = []
            for i, fileno in enumerate(
                range(file_start+2*j, file_stop+2*j+1, 2*len(unique_params))
            ):
                sr, yag_on = read_time_of_flight_from_zip(
                    get_zip_archive(
                        self.rootpath, self.year, self.month, self.day,
                        fileno, self.prefix
                    )
                )
                _, yag_off = read_time_of_flight_from_zip(
                    get_zip_archive(
                        self.rootpath, self.year, self.month, self.day,
                        fileno+1, self.prefix
                    )
                )
                if len(yag_off) == len(yag_on):
                    _tof_array.append(yag_on - yag_off)
                else:
                    print(f"File Error at {fileno}, {fileno+1}")
            tof_array: np.ndarray = np.array(_tof_array)
            tofs[j, :] = tof_array.mean(axis=0)
            _n: np.ndarray = np.sum(
                tof_array[:, bin_start:bin_end],
                axis=(1)
            )
            n[j] = _n.mean()
            dn[j] = _n.std()/np.sqrt(i+1)

        if "xscale" in kwargs:
            xscale = kwargs["xscale"]
        else:
            xscale = 1.0
        if "xoffset" in kwargs:
            xoffset = kwargs["xoffset"]
        else:
            xoffset = 0.0

        params = xscale*(np.array(unique_params)-xoffset)
        params_excluded = np.delete(params, param_index_fit_exclude)
        n_excluded = np.delete(n, param_index_fit_exclude)
        dn_excluded = np.delete(dn, param_index_fit_exclude)
        t = np.arange(0, 1000, 1)/sr

        fit = self._1D_fit(
            fitting,
            params_excluded,
            n_excluded,
            dn_excluded
        )
        _, ax = self._1D_plot(
            file_start, file_stop,
            params, y_mean=n, y_err=dn,
            fit=fit,
            **kwargs
        )
        if "display_images" in kwargs:
            if kwargs["display_images"]:
                i = 0
                for _tof in tofs:
                    _, ax = plt.subplots(1, 1, figsize=(8, 5))
                    ax.plot(t, _tof)
                    ax.set_title(f"{parameter}: {unique_params[i]:.3f}")
                    i += 1
                    if "tof_ylim" in kwargs:
                        ax.set_ylim(kwargs["tof_ylim"])
                    if "tof_xlim" in kwargs:
                        ax.set_xlim(kwargs["tof_xlim"])
        return fit, tofs

    def temperature(
        self,
        file_start: int,
        file_stop: int,
        parameter: str,
        row_start: int = 0,
        row_end: int = -1,
        col_start: int = 0,
        col_end: int = -1,
        fitting: str = None,
        param_index_fit_exclude: List[int] = [],
        **kwargs
    ) -> Tuple[Fit, Fit, np.ndarray]:
        unique_params, data_dict = self.get_unique_parameters(
            file_start, file_stop, parameter
        )
        _imgs = read_images_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month, self.day,
                file_start, self.prefix
            )
        )
        imgs = np.zeros(
            (len(unique_params), _imgs.shape[-2], _imgs.shape[-1]),
            dtype=float
        )
        h_width = np.zeros(len(unique_params))
        v_width = np.zeros(len(unique_params))
        if "display_fits" in kwargs:
            display_fits = kwargs["display_fits"]
        else:
            display_fits = None
        for j in range(len(unique_params)):
            _img_array: List[np.ndarray] = []
            for i, fileno in enumerate(
                range(file_start+2*j, file_stop+2*j+1, 2*len(unique_params))
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
                _img_array.append(np.mean(yag_on[1:] - yag_off[1:], axis=0))
            img_array: np.ndarray = np.array(_img_array)
            imgs[j, :, :] = img_array.mean(axis=0)
            _summed_img = img_array.sum(axis=0)
            v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                _summed_img[col_start:col_end, row_start:row_end],
                pixel_size=16e-6, bin_size=4, magnification=0.7
            )
            h_width[j] = h_fit.width
            v_width[j] = v_fit.width
            if display_fits:
                fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
                fig.subplots_adjust(hspace=0.01)
                ax[0].plot(
                    h_fit.x, h_fit.y, ".k",
                    h_fit.x_fine, h_fit.y_fine, "-r"
                )
                ax[1].plot(
                    v_fit.x, v_fit.y, ".k",
                    v_fit.x_fine, v_fit.y_fine, "-r"
                )
                ax[0].set_ylabel("Horizotal")
                ax[1].set_ylabel("Vertical")
                ax[1].set_xlabel("Spatial dimension")
                ax[0].set_title(f"{parameter}: {unique_params[j]}")

        unique_params = np.array(unique_params)*1e-5
        unique_params_excluded = np.delete(
            unique_params,
            param_index_fit_exclude
        )
        h_width_excluded = np.delete(h_width, param_index_fit_exclude)
        v_width_excluded = np.delete(v_width, param_index_fit_exclude)
        h_slope_fit = fit_linear(
            unique_params_excluded**2,
            h_width_excluded**2
        )
        v_slope_fit = fit_linear(
            unique_params_excluded**2,
            v_width_excluded**2
        )
        h_temp = h_slope_fit.slope*(59*cn.u)/cn.k
        v_temp = v_slope_fit.slope*(59*cn.u)/cn.k
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        fig.subplots_adjust(wspace=0.01)
        ax[0].plot(
            h_slope_fit.x*1e6, h_slope_fit.y*1e6, "ok",
            h_slope_fit.x_fine*1e6, h_slope_fit.y_fine*1e6, "-r"
        )
        ax[1].plot(
            v_slope_fit.x*1e6, v_slope_fit.y*1e6, "ok",
            v_slope_fit.x_fine*1e6, v_slope_fit.y_fine*1e6, "-r"
        )
        ax[0].set_title(f"T Horiz. : {h_temp*1e6:.2f} uK")
        ax[1].set_title(f"T Vert.  : {v_temp*1e6:.2f} uK")
        ax[0].set_xlabel("Sq. time [ms^2]")
        ax[1].set_xlabel("Sq. time [ms^2]")
        ax[0].set_ylabel("Sq. distance [mm^2]")
        ax[1].set_ylabel("Sq. distance [mm^2]")
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()

        if "display_images" in kwargs:
            if kwargs["display_images"]:
                if "image_figsize" in kwargs:
                    figsize = kwargs["image_figsize"]
                else:
                    figsize = (4, 4)
                for j in range(len(unique_params)):
                    _, ax = plt.subplots(1, 1, figsize=figsize)
                    _im = ax.imshow(imgs[j, :, :])
                    ax.set_title(f"{parameter}: {unique_params[j]:.3f}")
                    if "clim" in kwargs:
                        _im.set_clim(kwargs["clim"])
                    if "grid" in kwargs:
                        ax.grid(kwargs["grid"])
                    if "show_roi" in kwargs:
                        if kwargs["show_roi"]:
                            ax.add_patch(
                                Rectangle(
                                    (row_start, col_start),
                                    row_end-row_start, col_end-col_start,
                                    edgecolor='white',
                                    facecolor='none',
                                    fill=False,
                                    lw=1
                                )
                            )
        return v_slope_fit, h_slope_fit, imgs

    def number_by_image_2d(
        self,
        file_start: int,
        file_stop: int,
        parameters: List[str],
        row_start: int = 0,
        row_end: int = -1,
        col_start: int = 0,
        col_end: int = -1,
        fitting: str = None,
        param_index_fit_exclude: List[int] = [],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        unique_params1, data_dict = self.get_unique_parameters(
            file_start, file_stop, parameters[0]
        )
        _, data_dict1 = self.get_unique_parameters(
            file_start, file_start, parameters[0]
        )
        _, data_dict2 = self.get_unique_parameters(
            file_start+2, file_start+2, parameters[0]
        )

        if data_dict1[parameters[0]] != data_dict2[parameters[0]]:
            unique_params2, data_dict = self.get_unique_parameters(
                file_start, file_stop, parameters[1]
            )
        else:
            unique_params2 = unique_params1
            unique_params1, data_dict = self.get_unique_parameters(
                file_start, file_stop, parameters[1]
            )
            parameters[::-1]

        _imgs = read_images_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month, self.day,
                file_start, self.prefix
            )
        )
        imgs = np.zeros(
            (
                len(unique_params2),
                len(unique_params1),
                _imgs.shape[-2],
                _imgs.shape[-1]
            ),
            dtype=float
        )
        n = np.zeros((len(unique_params2), len(unique_params1)))
        dn = np.zeros((len(unique_params2), len(unique_params1)))
        exposure_time = data_dict[self.exposure_time_param]*1e-5
        number_multiplier = self.photon/(
            exposure_time*self.constants["gamma"]
            * self.constants["collection_solid_angle"]
        )
        tally = []
        for j in range(len(unique_params2)):
            for k in range(len(unique_params1)):
                _img_array: List[np.ndarray] = []
                for i, fileno in enumerate(
                    range(
                        file_start+2*len(unique_params1)*j+2*k,
                        file_stop+2*len(unique_params1)*j+2*k+1,
                        2*len(unique_params1)*len(unique_params2)
                    )
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
                    _img_array.append(
                        np.mean(yag_on[1:] - yag_off[1:], axis=0)
                    )
                img_array: np.ndarray = np.array(_img_array)
                imgs[j, k, :, :] = img_array.mean(axis=0)
                _n: np.ndarray = number_multiplier*np.sum(
                    img_array[:, col_start:col_end, row_start:row_end],
                    axis=(1, 2)
                )
                n[j, k] = _n.mean()
                dn[j, k] = _n.std()/np.sqrt(i+1)
                tally.append([unique_params1[k], unique_params2[j], n])

        if "xscale" in kwargs:
            xscale = kwargs["xscale"]
        else:
            xscale = 1.0
        if "xoffset" in kwargs:
            xoffset = kwargs["xoffset"]
        else:
            xoffset = 0.0
        if "yscale" in kwargs:
            yscale = kwargs["yscale"]
        else:
            yscale = 1.0
        if "yoffset" in kwargs:
            yoffset = kwargs["yoffset"]
        else:
            yoffset = 0.0
        if "xlabel" in kwargs:
            xlabel = kwargs["xlabel"]
        else:
            xlabel = parameters[0]
        if "ylabel" in kwargs:
            ylabel = kwargs["ylabel"]
        else:
            ylabel = parameters[1]
        if "figsize" in kwargs:
            figsize = kwargs["figsize"]
        else:
            figsize = (8, 5)
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        im = ax[0].imshow(
            n,
            extent=(
                (unique_params1[0]-xoffset)*xscale,
                (unique_params1[-1]-xoffset)*xscale,
                (unique_params2[-1]-yoffset)*yscale,
                (unique_params2[0]-yoffset)*yscale
            )
        )
        ax[1].imshow(
            dn,
            extent=(
                (unique_params1[0]-xoffset)*xscale,
                (unique_params1[-1]-xoffset)*xscale,
                (unique_params2[-1]-yoffset)*yscale,
                (unique_params2[0]-yoffset)*yscale
            )
        )
        fig.colorbar(im, ax=ax[0])
        for iax in ax:
            iax.set_xlabel(xlabel)
            iax.set_ylabel(ylabel)
            iax.grid(False)
        return imgs, n, dn, tally

    def number_by_image_2d_bgImage(
        self,
        file_start: int,
        file_stop: int,
        parameters: List[str],
        row_start: int = 0,
        row_end: int = -1,
        col_start: int = 0,
        col_end: int = -1,
        fitting: str = None,
        param_index_fit_exclude: List[int] = [],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        unique_params1, data_dict = self.get_unique_parameters_bgImage(
            file_start, file_stop, parameters[0]
        )
        _, data_dict1 = self.get_unique_parameters_bgImage(
            file_start, file_start, parameters[0]
        )
        _, data_dict2 = self.get_unique_parameters_bgImage(
            file_start+1, file_start+1, parameters[0]
        )

        if data_dict1[parameters[0]] != data_dict2[parameters[0]]:
            unique_params2, data_dict = self.get_unique_parameters_bgImage(
                file_start, file_stop, parameters[1]
            )
        else:
            unique_params2 = unique_params1
            unique_params1, data_dict = self.get_unique_parameters_bgImage(
                file_start, file_stop, parameters[1]
            )
            parameters[::-1]

        _imgs = read_images_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month, self.day,
                file_start, self.prefix
            )
        )
        imgs = np.zeros(
            (
                len(unique_params2),
                len(unique_params1),
                _imgs.shape[-2],
                _imgs.shape[-1]
            ),
            dtype=float
        )
        n = np.zeros((len(unique_params2), len(unique_params1)))
        dn = np.zeros((len(unique_params2), len(unique_params1)))
        exposure_time = data_dict[self.exposure_time_param]*1e-5
        number_multiplier = self.photon/(
            exposure_time*self.constants["gamma"]
            * self.constants["collection_solid_angle"]
        )
        tally = []
        for j in range(len(unique_params2)):
            for k in range(len(unique_params1)):
                _img_array: List[np.ndarray] = []
                for i, fileno in enumerate(
                    range(
                        file_start+len(unique_params1)*j+k,
                        file_stop+len(unique_params1)*j+k,
                        len(unique_params1)*len(unique_params2)
                    )
                ):
                    yag_on = read_images_from_zip(
                        get_zip_archive(
                            self.rootpath, self.year, self.month, self.day,
                            fileno, self.prefix
                        )
                    )
                    _img_array.append(
                        np.mean(yag_on[::2] - yag_on[1::2], axis=0)
                    )
                img_array: np.ndarray = np.array(_img_array)
                imgs[j, k, :, :] = img_array.mean(axis=0)
                _n: np.ndarray = number_multiplier*np.sum(
                    img_array[:, col_start:col_end, row_start:row_end],
                    axis=(1, 2)
                )
                n[j, k] = _n.mean()
                dn[j, k] = _n.std()/np.sqrt(i+1)
                tally.append([unique_params1[k], unique_params2[j], n[j, k]])

        if "xscale" in kwargs:
            xscale = kwargs["xscale"]
        else:
            xscale = 1.0
        if "xoffset" in kwargs:
            xoffset = kwargs["xoffset"]
        else:
            xoffset = 0.0
        if "yscale" in kwargs:
            yscale = kwargs["yscale"]
        else:
            yscale = 1.0
        if "yoffset" in kwargs:
            yoffset = kwargs["yoffset"]
        else:
            yoffset = 0.0
        if "xlabel" in kwargs:
            xlabel = kwargs["xlabel"]
        else:
            xlabel = parameters[0]
        if "ylabel" in kwargs:
            ylabel = kwargs["ylabel"]
        else:
            ylabel = parameters[1]
        if "figsize" in kwargs:
            figsize = kwargs["figsize"]
        else:
            figsize = (8, 5)
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        im = ax[0].imshow(
            n,
            extent=(
                (unique_params1[0]-xoffset)*xscale,
                (unique_params1[-1]-xoffset)*xscale,
                (unique_params2[-1]-yoffset)*yscale,
                (unique_params2[0]-yoffset)*yscale
            )
        )
        ax[1].imshow(
            dn,
            extent=(
                (unique_params1[0]-xoffset)*xscale,
                (unique_params1[-1]-xoffset)*xscale,
                (unique_params2[-1]-yoffset)*yscale,
                (unique_params2[0]-yoffset)*yscale
            )
        )
        fig.colorbar(im, ax=ax[0])
        for iax in ax:
            iax.set_xlabel(xlabel)
            iax.set_ylabel(ylabel)
            iax.grid(False)
        return imgs, n, dn, tally
    
    def shape_by_image_2d_bgImage(
        self,
        file_start: int,
        file_stop: int,
        parameters: List[str],
        n_combine: int = 1,
        row_start: int = 0,
        row_end: int = -1,
        col_start: int = 0,
        col_end: int = -1,
        fitting: str = None,
        param_index_fit_exclude: List[int] = [],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        unique_params1, data_dict = self.get_unique_parameters_bgImage(
            file_start, file_stop, parameters[0]
        )
        _, data_dict1 = self.get_unique_parameters_bgImage(
            file_start, file_start, parameters[0]
        )
        _, data_dict2 = self.get_unique_parameters_bgImage(
            file_start+1, file_start+1, parameters[0]
        )

        if data_dict1[parameters[0]] != data_dict2[parameters[0]]:
            unique_params2, data_dict = self.get_unique_parameters_bgImage(
                file_start, file_stop, parameters[1]
            )
        else:
            unique_params2 = unique_params1
            unique_params1, data_dict = self.get_unique_parameters_bgImage(
                file_start, file_stop, parameters[1]
            )
            parameters[::-1]

        _imgs = read_images_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month, self.day,
                file_start, self.prefix
            )
        )
        n_iter = len(range(file_start, file_stop+1, len(unique_params2)*len(unique_params1)))
        n_sets = int(n_iter/n_combine)
        imgs = np.zeros(
            (
                len(unique_params2),
                len(unique_params1),
                n_iter,
                _imgs.shape[-2],
                _imgs.shape[-1]
            ),
            dtype=float
        )
        
        n = np.zeros((len(unique_params2), len(unique_params1)))
        dn = np.zeros((len(unique_params2), len(unique_params1)))
        exposure_time = data_dict[self.exposure_time_param]*1e-5
        number_multiplier = self.photon/(
            exposure_time*self.constants["gamma"]
            * self.constants["collection_solid_angle"]
        )
        horizontal_width = np.zeros((len(unique_params2),len(unique_params1), n_sets))
        horizontal_centre = np.zeros((len(unique_params2),len(unique_params1), n_sets))
        vertical_width = np.zeros((len(unique_params2),len(unique_params1), n_sets))
        vertical_centre = np.zeros((len(unique_params2),len(unique_params1), n_sets))
        tally = []
        for j in range(len(unique_params2)):
            for k in range(len(unique_params1)):
                _img_array: List[np.ndarray] = []
                for i, fileno in enumerate(
                    range(
                        file_start+len(unique_params1)*j+k,
                        file_stop+len(unique_params1)*j+k,
                        len(unique_params1)*len(unique_params2)
                    )
                ):
                    yag_on = read_images_from_zip(
                        get_zip_archive(
                            self.rootpath, self.year, self.month, self.day,
                            fileno, self.prefix
                        )
                    )
                    _img_array.append(
                        np.mean(yag_on[::2] - yag_on[1::2], axis=0)
                    )
                img_array: np.ndarray = np.array(_img_array)
                imgs[j, k, :, :, :] = img_array
                for i in range(n_sets):
                    try:
                        _img = imgs[
                                j, k, i*n_combine:(i+1)*n_combine, :, :
                            ].sum(axis=0)
                        h_fit, v_fit = \
                            calculate_cloud_size_from_image_1d_gaussian(
                                _img,
                                pixel_size=self.constants["pixel_size"],
                                bin_size=self.constants["binning"],
                                magnification=self.constants["magnification"]
                            )
                        #self.horizontal_fits[f"({i}, {j})"] = h_fit
                        #self.vertical_fits[f"({i}, {j})"] = v_fit
                        if h_fit is not None:
                            horizontal_width[j, k, i] = h_fit.width
                            horizontal_centre[j, k, i] = h_fit.centre
                        if v_fit is not None:
                            vertical_width[j, k, i] = v_fit.width
                            vertical_centre[j, k, i] = v_fit.centre
                    except Exception as e:
                        file_no = file_start+j*len(unique_params1)+k+i*n_iter
                        print(
                            f"Error {e} occured in file {file_no} in size fit"
                        )
                tally.append([unique_params1[k], unique_params2[j], np.array(horizontal_width[j,k,:]).mean()])
        horizontal_width_mean = 1e3*np.array(horizontal_width).mean(axis=2)
        horizontal_centre_mean = 1e3*np.array(horizontal_centre).mean(axis=2)
        vertical_width_mean = 1e3*np.array(vertical_width).mean(axis=2)
        vertical_centre_mean = 1e3*np.array(vertical_centre).mean(axis=2)

        if "xscale" in kwargs:
            xscale = kwargs["xscale"]
        else:
            xscale = 1.0
        if "xoffset" in kwargs:
            xoffset = kwargs["xoffset"]
        else:
            xoffset = 0.0
        if "yscale" in kwargs:
            yscale = kwargs["yscale"]
        else:
            yscale = 1.0
        if "yoffset" in kwargs:
            yoffset = kwargs["yoffset"]
        else:
            yoffset = 0.0
        if "xlabel" in kwargs:
            xlabel = kwargs["xlabel"]
        else:
            xlabel = parameters[0]
        if "ylabel" in kwargs:
            ylabel = kwargs["ylabel"]
        else:
            ylabel = parameters[1]
        if "figsize" in kwargs:
            figsize = kwargs["figsize"]
        else:
            figsize = (8, 8)
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        im00 = ax[0][0].imshow(
            horizontal_width_mean,
            extent=(
                (unique_params1[0]-xoffset)*xscale,
                (unique_params1[-1]-xoffset)*xscale,
                (unique_params2[-1]-yoffset)*yscale,
                (unique_params2[0]-yoffset)*yscale
            )
        )
        ax[0][0].set_title("horizontal width")
        im01 = ax[0][1].imshow(
            vertical_width_mean,
            extent=(
                (unique_params1[0]-xoffset)*xscale,
                (unique_params1[-1]-xoffset)*xscale,
                (unique_params2[-1]-yoffset)*yscale,
                (unique_params2[0]-yoffset)*yscale
            )
        )
        ax[0][1].set_title("vertical width")
        im10 = ax[1][0].imshow(
            horizontal_centre_mean,
            extent=(
                (unique_params1[0]-xoffset)*xscale,
                (unique_params1[-1]-xoffset)*xscale,
                (unique_params2[-1]-yoffset)*yscale,
                (unique_params2[0]-yoffset)*yscale
            )
        )
        ax[1][0].set_title("horizontal centre")
        im11 = ax[1][1].imshow(
            vertical_centre_mean,
            extent=(
                (unique_params1[0]-xoffset)*xscale,
                (unique_params1[-1]-xoffset)*xscale,
                (unique_params2[-1]-yoffset)*yscale,
                (unique_params2[0]-yoffset)*yscale
            )
        )
        ax[1][1].set_title("vertical centre")
        fig.colorbar(im00, ax=ax[0][0], location='right', shrink=0.8)
        fig.colorbar(im01, ax=ax[0][1], location='right', shrink=0.8)
        fig.colorbar(im10, ax=ax[1][0], location='right', shrink=0.8)
        fig.colorbar(im11, ax=ax[1][1], location='right', shrink=0.8)
        for iax in ax.flatten():
            iax.set_xlabel(xlabel)
            iax.set_ylabel(ylabel)
            iax.grid(False)
        fig.tight_layout()
        return imgs, horizontal_width_mean, horizontal_centre_mean, vertical_width_mean, vertical_centre_mean, tally

    def position_by_image(
        self,
        file_start: int,
        file_stop: int,
        parameter: str,
        row_start: int = 0,
        row_end: int = -1,
        col_start: int = 0,
        col_end: int = -1,
        fitting: str = None,
        param_index_fit_exclude: List[int] = [],
        **kwargs
    ) -> Tuple[Fit, Fit, np.ndarray]:
        unique_params, data_dict = self.get_unique_parameters(
            file_start, file_stop, parameter
        )
        _imgs = read_images_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month, self.day,
                file_start, self.prefix
            )
        )
        imgs = np.zeros(
            (len(unique_params), _imgs.shape[-2], _imgs.shape[-1]),
            dtype=float
        )
        h_centre = np.zeros(len(unique_params))
        v_centre = np.zeros(len(unique_params))
        if "display_fits" in kwargs:
            display_fits = kwargs["display_fits"]
        else:
            display_fits = None
        for j in range(len(unique_params)):
            _img_array: List[np.ndarray] = []
            for i, fileno in enumerate(
                range(file_start+2*j, file_stop+2*j+1, 2*len(unique_params))
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
                _img_array.append(np.mean(yag_on - yag_off, axis=0))
            img_array: np.ndarray = np.array(_img_array)
            imgs[j, :, :] = img_array.mean(axis=0)
            v_fit, h_fit = calculate_cloud_size_from_image_1d_gaussian(
                imgs[j, col_start:col_end, row_start:row_end],
                pixel_size=self.constants["pixel_size"],
                bin_size=self.constants["binning"],
                magnification=self.constants["magnification"]
            )
            v_centre[j] = v_fit.centre
            h_centre[j] = h_fit.centre
            if display_fits:
                fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
                fig.subplots_adjust(hspace=0.01)
                ax[0].plot(
                    h_fit.x, h_fit.y, ".k",
                    h_fit.x_fine, h_fit.y_fine, "-r"
                )
                ax[1].plot(
                    v_fit.x, v_fit.y, ".k",
                    v_fit.x_fine, v_fit.y_fine, "-r"
                )
                ax[0].set_ylabel("Horizotal")
                ax[1].set_ylabel("Vertical")
                ax[1].set_xlabel("Spatial dimension")
                ax[0].set_title(f"{parameter}: {unique_params[j]}")

        if "xscale" in kwargs:
            xscale = kwargs["xscale"]
        else:
            xscale = 1.0
        if "xoffset" in kwargs:
            xoffset = kwargs["xoffset"]
        else:
            xoffset = 0.0
        if "yscale" in kwargs:
            yscale = kwargs["yscale"]
        else:
            yscale = 1.0
        if "yoffset" in kwargs:
            yoffset = kwargs["yoffset"]
        else:
            yoffset = 0.0

        params = xscale*(np.array(unique_params)-xoffset)
        v_centre = yscale*(v_centre-yoffset)
        h_centre = yscale*(h_centre-yoffset)
        params_excluded = np.delete(params, param_index_fit_exclude)
        h_centre_excluded = np.delete(h_centre, param_index_fit_exclude)
        v_centre_excluded = np.delete(v_centre, param_index_fit_exclude)

        h_centre_fit = self._1D_fit(
            fitting,
            params_excluded,
            h_centre_excluded
        )
        v_centre_fit = self._1D_fit(
            fitting,
            params_excluded,
            v_centre_excluded
        )

        _, ax = self._1D_plot(
            file_start, file_stop,
            params, y_mean=h_centre, y_err=None,
            fit=h_centre_fit,
            title="Horizontal position",
            **kwargs
        )
        _, ax = self._1D_plot(
            file_start, file_stop,
            params, y_mean=v_centre, y_err=None,
            fit=v_centre_fit,
            title="Vertical position",
            **kwargs
        )
        if "display_images" in kwargs:
            if kwargs["display_images"]:
                if "image_figsize" in kwargs:
                    figsize = kwargs["image_figsize"]
                else:
                    figsize = (4, 4)
                for j in range(len(unique_params)):
                    _, ax = plt.subplots(1, 1, figsize=figsize)
                    _im = ax.imshow(imgs[j, :, :])
                    ax.set_title(f"{parameter}: {unique_params[j]}")
                    if "clim" in kwargs:
                        _im.set_clim(kwargs["clim"])
                    if "grid" in kwargs:
                        ax.grid(kwargs["grid"])
                    if "show_roi" in kwargs:
                        if kwargs["show_roi"]:
                            ax.add_patch(
                                Rectangle(
                                    (row_start, col_start),
                                    row_end-row_start, col_end-col_start,
                                    edgecolor='white',
                                    facecolor='none',
                                    fill=False,
                                    lw=1
                                )
                            )
        return v_centre_fit, h_centre_fit, imgs
