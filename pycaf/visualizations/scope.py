from typing import Tuple, List
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import json

from ..analysis.models import Fit
from ..analysis import (
    get_zip_archive,
    read_time_of_flight_from_zip,
    read_time_of_flight_from_zip_no_mean,
    read_images_from_zip,
    read_parameters_from_zip,
    smooth_time_of_flight,
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


class Scope():
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
        if fit:
            ax.plot(fit.x_fine, fit.y_fine, "-r", label="Fit")
            ax.text(
                1.03, 0.98,
                "Fitting info:\n"+fit.func_str+fit.args_str,
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
        ax.legend()
        return fig, ax

    def _1D_fit(
        self,
        fitting: str,
        x: np.ndarray,
        y_mean: np.ndarray,
        y_err: np.ndarray,
    ) -> Fit:
        fit = None
        if fitting in fitting_func_map:
            fit = fitting_func_map[fitting](
                    x,
                    y_mean,
                    y_err
                )
        return fit

    def multishot_altfile_background(
        self,
        file_start: int,
        file_stop: int,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        _all_params = read_parameters_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month, self.day,
                file_start, self.prefix
            )
        )
        exposure_time = _all_params[self.exposure_time_param]*1e-5
        number_multiplier = self.photon/(
            exposure_time*self.constants["gamma"]
            * self.constants["collection_solid_angle"]
        )
        n, img, tof = [], [], []
        for fileno in range(file_start, file_stop+1, 2):
            img_yag_on = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            img_yag_off = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno+1, self.prefix
                )
            )
            sr, tof_yag_on = read_time_of_flight_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            sr, tof_yag_off = read_time_of_flight_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno+1, self.prefix
                )
            )
            _img = np.mean(img_yag_on-img_yag_off, axis=0)
            _tof = tof_yag_on - tof_yag_off
            _n = number_multiplier*np.sum(
                _img[row_start:row_end, col_start:col_end]
            )
            n.append(_n)
            img.append(_img)
            tof.append(_tof)
        img = np.mean(img, axis=0)
        tof = np.mean(tof, axis=0)
        t = 1000*np.arange(0, 1000)*(1.0/sr)
        h_profile = np.sum(img, axis=0)
        v_profile = np.sum(img, axis=1)
        h, w = len(h_profile), len(v_profile)
        ratio = h / float(w)
        x = self.scale_multiplier*np.arange(0, len(h_profile), 1)
        y = self.scale_multiplier*np.arange(0, len(v_profile), 1)
        row_start *= self.scale_multiplier
        row_end *= self.scale_multiplier
        col_start *= self.scale_multiplier
        col_end *= self.scale_multiplier
        h_fit = fit_gaussian_with_offset(y, h_profile)
        v_fit = fit_gaussian_with_offset(x, v_profile)

        width_ax0 = 4.0
        width_ax1 = 1.0
        height_ax2 = 1.0
        width_ax3 = 5.0
        height_ax0 = width_ax0 * ratio

        left_margin = 0.65
        right_margin = 0.2
        bottom_margin = 0.5
        top_margin = 0.25
        inter_margin = 0.1

        fwidth = left_margin + right_margin + inter_margin \
            + width_ax0 + width_ax1 + width_ax3
        fheight = bottom_margin + top_margin + inter_margin \
            + height_ax0 + height_ax2

        fig = plt.figure(figsize=(fwidth, fheight))
        fig.patch.set_facecolor('white')

        ax0 = fig.add_axes(
            [
                left_margin / fwidth,
                (bottom_margin + inter_margin + height_ax2) / fheight,
                width_ax0 / fwidth,
                height_ax0 / fheight
            ]
        )
        ax1 = fig.add_axes(
            [
                (left_margin + width_ax0 + inter_margin) / fwidth,
                (bottom_margin + inter_margin + height_ax2) / fheight,
                width_ax1 / fwidth,
                height_ax0 / fheight
            ]
        )
        ax2 = fig.add_axes(
            [
                left_margin / fwidth,
                bottom_margin / fheight,
                width_ax0 / fwidth,
                height_ax2 / fheight
            ]
        )
        ax3 = fig.add_axes(
            [
                (left_margin + width_ax0 + inter_margin
                 + width_ax1 + 4*inter_margin) / fwidth,
                (bottom_margin + inter_margin + height_ax2) / fheight,
                width_ax3 / fwidth, height_ax0 / fheight
            ]
        )
        ax4 = fig.add_axes(
            [
                (left_margin + width_ax0 + inter_margin) / fwidth,
                (bottom_margin) / fheight,
                (width_ax1 + width_ax3 + 4*inter_margin) / fwidth,
                (height_ax2 - 6*inter_margin) / fheight
            ]
        )

        bounds = [x.min(), x.max(), y.min(), y.max()]
        _img = ax0.imshow(img, extent=bounds, origin='lower')
        if "clim" in kwargs:
            _img.set_clim(kwargs["clim"])
        fig.colorbar(_img, cax=ax4, orientation="horizontal")
        ax0.add_patch(
            Rectangle(
                (row_start, col_start), row_end-row_start, col_end-col_start,
                edgecolor='white',
                facecolor='none',
                fill=False,
                lw=1
            )
        )
        ax0.grid(False)
        ax0.set_yticks([])
        ax0.set_xticks([])
        n_str = f"N = {np.mean(n):.0f}+/-{np.std(n)/np.sqrt(len(n)):.0f}"
        ax0.text(0.5, 0.5, n_str, color="white", fontsize=15)
        ax0.set_ylabel("Distance [mm]")
        ax1.plot(v_profile, x, ".")
        ax1.plot(
            v_fit.y_fine,
            v_fit.x_fine,
            "-r", label=f"{v_fit.width:.2f} mm"
        )
        ax1.invert_xaxis()
        ax1.yaxis.tick_right()
        ax1.legend()
        ax2.plot(y, h_profile, ".")
        ax2.plot(
            h_fit.x_fine,
            h_fit.y_fine,
            "-r", label=f"{h_fit.width:.2f} mm"
        )
        ax2.set_xlabel("Distance [mm]")
        ax2.legend()
        ax3.plot(t, tof, ".k", t, smooth_time_of_flight(tof), "-r")
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        ax3.set_ylabel("PMT signal [V]")
        ax3.set_xlabel("Time [ms]")
        plt.show(block=False)
        return fig, (ax0, ax1, ax2, ax3, ax4)

    def multishot_altfile_background_tof(
        self,
        file_start: int,
        file_stop: int,
        bin_start: int = 0,
        bin_stop: int = -1,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
        tofs = []
        for fileno in range(file_start, file_stop+1, 2):
            sr, tof_yag_on = read_time_of_flight_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            _, tof_yag_off = read_time_of_flight_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno+1, self.prefix
                )
            )
            _tof = tof_yag_on-tof_yag_off
            tofs.append(_tof)
        tofs = np.array(tofs)
        tofs_max = np.max(tofs, axis=0)
        tofs_min = np.min(tofs, axis=0)
        tofs = tofs.mean(axis=0)
        int_val = np.sum(tofs[bin_start: bin_stop])
        t = np.arange(0, 1000, 1)*(1000.0/sr)

        fig, ax = self._1D_plot(
            t, y_mean=tofs,
            label=f"Sum(Bin region): {int_val:.2f}",
            **kwargs
        )
        if "show_variation" in kwargs:
            if kwargs["show_variation"]:
                ax.fill_between(
                    t,
                    tofs_min,
                    tofs_max,
                    alpha=0.1,
                    color="k"
                )
        if "show_bin" in kwargs:
            if kwargs["show_bin"]:
                if "facecolor" in kwargs:
                    fc = kwargs["facecolor"]
                else:
                    fc = "blue"
                if "alpha" in kwargs:
                    alpha = kwargs["alpha"]
                else:
                    alpha = 0.2
                ax.fill_between(
                    t[bin_start: bin_stop],
                    tofs[bin_start: bin_stop],
                    alpha=alpha,
                    color=fc
                )
        return fig, ax, tofs

    def multishot_altfile_background_parameter_variation_with_image(
        self,
        file_start: int,
        file_stop: int,
        parameter: str,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        fitting: str = None,
        param_index_fit_exclude: List = [],
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, Fit, np.ndarray]:
        _all_params = read_parameters_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month, self.day,
                file_start, self.prefix
            )
        )
        exposure_time = _all_params[self.exposure_time_param]*1e-5
        number_multiplier = self.photon/(
            exposure_time*self.constants["gamma"]
            * self.constants["collection_solid_angle"]
        )
        n, img, params = [], [], []
        for fileno in range(file_start, file_stop+1, 2):
            img_yag_on = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            img_yag_off = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno+1, self.prefix
                )
            )
            _all_params = read_parameters_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno+1, self.prefix
                )
            )
            _, img_rows, img_cols = img_yag_off.shape
            _img = np.mean(img_yag_on-img_yag_off, axis=0)
            _n = number_multiplier*np.sum(
                _img[row_start:row_end, col_start:col_end]
            )
            n.append(_n)
            img.append(_img)
            params.append(_all_params[parameter])

        for i, param in enumerate(params):
            if i != 0 and param == params[0]:
                len_of_params = i
                break
        assert (file_stop+1-file_start) % len_of_params == 0

        if "xscale" in kwargs:
            xscale = kwargs["xscale"]
        else:
            xscale = 1.0
        if "xoffset" in kwargs:
            xoffset = kwargs["xoffset"]
        else:
            xoffset = 0.0
        params = np.array(params[:len_of_params])*xscale-xoffset
        rep = int((file_stop+1-file_start) / (2*len_of_params))
        n = np.array(n).reshape((rep, len_of_params))
        n_mean = np.mean(n, axis=0)
        n_err = np.std(n, axis=0)/np.sqrt(rep)
        img = np.array(img).reshape((rep, len_of_params, img_rows, img_cols))
        img = np.mean(img, axis=0)

        params_excluded = np.delete(params, param_index_fit_exclude)
        n_mean_excluded = np.delete(n_mean, param_index_fit_exclude)
        n_err_excluded = np.delete(n_err, param_index_fit_exclude)

        fit = self._1D_fit(
            fitting,
            params_excluded,
            n_mean_excluded,
            n_err_excluded
        )
        fig, ax = self._1D_plot(
            params, y_mean=n_mean, y_err=n_err,
            fit=fit,
            **kwargs
        )
        return fig, ax, fit, img

    def multishot_altfile_background_parameter_variation_with_tof(
        self,
        file_start: int,
        file_stop: int,
        parameter: str,
        bin_start: int,
        bin_end: int,
        fitting: str = None,
        param_index_fit_exclude: List = [],
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, Fit, np.ndarray]:
        n, tofs, params = [], [], []
        for i, fileno in enumerate(range(file_start, file_stop+1, 2)):
            _, tof_yag_on = read_time_of_flight_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            _, tof_yag_off = read_time_of_flight_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno+1, self.prefix
                )
            )
            _all_params = read_parameters_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno+1, self.prefix
                )
            )
            _tof = tof_yag_on - tof_yag_off
            _n = np.sum(
                _tof[bin_start:bin_end]
            )
            n.append(_n)
            tofs.append(_tof)
            params.append(_all_params[parameter])

        for i, param in enumerate(params):
            if i != 0 and param == params[0]:
                len_of_params = i
                break
        assert (file_stop+1-file_start) % len_of_params == 0

        if "xscale" in kwargs:
            xscale = kwargs["xscale"]
        else:
            xscale = 1.0
        if "xoffset" in kwargs:
            xoffset = kwargs["xoffset"]
        else:
            xoffset = 0.0
        params = np.array(params[:len_of_params])*xscale-xoffset
        rep = int((file_stop+1-file_start) / (2*len_of_params))
        n = np.array(n).reshape((rep, len_of_params))
        n_mean = np.mean(n, axis=0)
        n_err = np.std(n, axis=0)/np.sqrt(rep)
        tofs = np.array(tofs).reshape((rep, len_of_params, 1000))
        tofs = np.mean(tofs, axis=0)

        params_excluded = np.delete(params, param_index_fit_exclude)
        n_mean_excluded = np.delete(n_mean, param_index_fit_exclude)
        n_err_excluded = np.delete(n_err, param_index_fit_exclude)

        fit = self._1D_fit(
            fitting,
            params_excluded,
            n_mean_excluded,
            n_err_excluded
        )
        fig, ax = self._1D_plot(
            params, y_mean=n_mean, y_err=n_err,
            fit=fit,
            **kwargs
        )
        return fig, ax, fit, tofs

    def multishot_altfile_background_frequency_variation_with_image(
        self,
        file_start: int,
        file_stop: int,
        frequencies: np.ndarray,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        fitting: str = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, Fit, np.ndarray]:
        _all_params = read_parameters_from_zip(
            get_zip_archive(
                self.rootpath, self.year, self.month, self.day,
                file_start, self.prefix
            )
        )
        exposure_time = _all_params[self.exposure_time_param]*1e-5
        number_multiplier = self.photon/(
            exposure_time*self.constants["gamma"]
            * self.constants["collection_solid_angle"]
        )
        n, err, img = [], [], []
        for fileno in range(file_start, file_stop+1, 2):
            img_yag_on = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            img_yag_off = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno+1, self.prefix
                )
            )
            _img = img_yag_on-img_yag_off
            _n = number_multiplier*np.sum(
                _img[:, row_start:row_end, col_start:col_end],
                axis=(1, 2)
            )
            n.append(_n.mean())
            img.append(_img.mean(axis=0))
            err.append(_n.std()/np.sqrt(len(_n)))
        img = np.array(img)
        n = np.array(n)
        err = np.array(err)
        fit = self._1D_fit(
            fitting,
            frequencies,
            n,
            err
        )
        fig, ax = self._1D_plot(
            frequencies, y_mean=n, y_err=err,
            fit=fit,
            **kwargs
        )
        return fig, ax, fit, img

    def multishot_altfile_background_frequency_variation_with_tof(
        self,
        file_start: int,
        file_stop: int,
        frequencies: np.ndarray,
        bin_start: int,
        bin_end: int,
        fitting: str = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, Fit, np.ndarray]:
        n, err, tofs = [], [], []
        for fileno in range(file_start, file_stop+1, 2):
            _, tof_yag_on = read_time_of_flight_from_zip_no_mean(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            _, tof_yag_off = read_time_of_flight_from_zip_no_mean(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno+1, self.prefix
                )
            )
            _tof = tof_yag_on-tof_yag_off
            _n = np.sum(
                _tof[:, bin_start:bin_end],
                axis=1
            )
            n.append(_n.mean())
            tofs.append(_tof.mean(axis=0))
            err.append(_n.std()/np.sqrt(len(_n)))
        tofs = np.array(tofs)
        n = np.array(n)
        err = np.array(err)
        fit = self._1D_fit(
            fitting,
            frequencies,
            n,
            err
        )
        fig, ax = self._1D_plot(
            frequencies, y_mean=n, y_err=err,
            fit=fit,
            **kwargs
        )
        return fig, ax, fit, tofs
