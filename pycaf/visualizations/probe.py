from typing import Tuple, List, Union, Dict
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
    calculate_cloud_size_from_image_1d_gaussian,
    fit_linear,
    fit_exponential_without_offset,
    fit_exponential_with_offset,
    fit_gaussian_with_offset,
    fit_gaussian_without_offset,
    fit_lorentzian_with_offset,
    fit_lorentzian_without_offset
)


fitting_func_map = \
    {
        "linear": fit_linear,
        "exponential_without_offset": fit_exponential_without_offset,
        "exponential_with_offset": fit_exponential_with_offset,
        "gaussian_without_offset": fit_gaussian_without_offset,
        "gaussian_with_offset": fit_gaussian_with_offset,
        "lorentzian_with_offset": fit_lorentzian_with_offset,
        "lorentzian_without_offset": fit_lorentzian_without_offset
    }


class Probe():
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

    def _display_all_image(
        self,
        img: np.ndarray,
        param: np.ndarray,
        **kwargs
    ) -> None:
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
                _tof_array.append(yag_on - yag_off)
            tof_array: np.ndarray = np.array(_tof_array)
            tofs[j, :] = tof_array.mean(axis=0)
            _n: np.ndarray = np.sum(
                tof_array[:, :, bin_start:bin_end],
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
                pixel_size=16e-6, bin_size=4, magnification=1.4
            )
            h_width[j] = h_fit.width
            v_width[j] = v_fit.width

        unique_params = np.array(unique_params)*1e-5
        h_slope_fit = fit_linear(unique_params**2, h_width**2)
        v_slope_fit = fit_linear(unique_params**2, v_width**2)
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
                    _img_array.append(np.mean(yag_on - yag_off, axis=0))
                img_array: np.ndarray = np.array(_img_array)
                imgs[j, k, :, :] = img_array.mean(axis=0)
                _n: np.ndarray = number_multiplier*np.sum(
                    img_array[:, col_start:col_end, row_start:row_end],
                    axis=(1, 2)
                )
                n[j, k] = _n.mean()
                dn[j, k] = _n.std()/np.sqrt(i+1)

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
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(
            n,
            extent=(
                (unique_params1[0]-xoffset)*xscale,
                (unique_params1[-1]-xoffset)*xscale,
                (unique_params2[-1]-yoffset)*yscale,
                (unique_params2[0]-yoffset)*yscale
            )
        )
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)
        return imgs, n
