from typing import Tuple, List, Union, Dict, Callable
from typing_extensions import Self
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import json
from PIL import Image
import scipy.constants as cn
from pathlib import Path
from scipy.stats import zscore, sem

from .analysis.utils import (
    month_dict
)

from .analysis.models import (
    Fit,
    LinearFit,
    GaussianFitWithOffset,
    ExponentialFitWithoutOffset
)

from .analysis import (
    get_zip_archive,
    read_time_of_flight_from_zip,
    read_images_from_zip,
    read_parameters_from_zip,
    read_frequencies_from_zip,
    read_gigatrons_from_zip,
    calculate_cloud_size_from_image_1d_gaussian,
    fit_linear,
    fit_quadratic_without_slope,
    fit_exponential_without_offset2,
    fit_exponential_with_offset2,
    fit_gaussian_with_offset,
    fit_gaussian_without_offset,
    fit_lorentzian_with_offset,
    fit_lorentzian_without_offset,
    fit_trap_frequency_oscillation
)

##### TO DO ######
## fit free fall
## show individual images with fits
## have the option of showing individual points on the plots, with colours indicating whether they are considered outliers or not

class Processor():
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
        self.outputrootpath = self.config["processed_root_path"]
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
        
        #### write this into a general header 
        #self.filepath = 
        #self.header_general = f" "
        
    def reset(self) -> Self:
        self.file_start: int = None
        self.file_stop: int = None
        self.number: np.ndarray = None
        self.horizontal_width: np.ndarray = None
        self.vertical_width: np.ndarray = None
        self.horizontal_centre: np.ndarray = None
        self.vertical_centre: np.ndarray = None
        self.density: np.ndarray = None
        return self

    def __call__(
        self,
        file_start: int,
        file_stop: int,
        parameter: Union[str, List[str]],
        normalisation: bool = False,
        show_images: bool = False
    ) -> Self:
        self.reset() # necessary once multiple steps of analysis are added?

        [data_dir, image_dir] = self.create_dir()
        
        ### check if csv file exists for these filenumbers and skip data reading otherwise?
        
        self.file_start = file_start
        self.file_stop = file_stop
        
        self.fileno = np.zeros(file_stop-file_start+1, dtype=np.uint16)
        self.number = np.zeros(file_stop-file_start+1, dtype=float)
        self.number1 = np.zeros(file_stop-file_start+1, dtype=float)
        self.number2 = np.zeros(file_stop-file_start+1, dtype=float)
        self.horizontal_width = np.zeros(file_stop-file_start+1)
        self.vertical_width = np.zeros(file_stop-file_start+1)
        self.horizontal_centre = np.zeros(file_stop-file_start+1)
        self.vertical_centre = np.zeros(file_stop-file_start+1)
        self.density = np.zeros(file_stop-file_start+1)
        
        if type(parameter) is str:
            parameter = [parameter]

        self.parameter = np.zeros((len(parameter), file_stop-file_start+1))
        self.parameter_names = parameter
        
        for i, fileno in enumerate(
            range(file_start, file_stop+1, 1)
        ):
            self.fileno[i] = fileno

            # get image(s) for file number
            _raw_images = read_images_from_zip(
                get_zip_archive(
                    self.rootpath, self.year, self.month, self.day,
                    fileno, self.prefix
                )
            )
            if normalisation:
                if len(_raw_images) % 3 == 0:
                    _image1 = np.mean(
                        _raw_images[0::3] - _raw_images[2::3],
                        axis=0
                    )
                    _image2 = np.mean(
                        _raw_images[1::3] - _raw_images[2::3],
                        axis=0
                    )
                else:
                    _image1 = np.zeros(np.shape(_raw_images[0]))
                    _image2 = np.zeros(np.shape(_raw_images[0]))
                self.save_image(_image2, fileno, image_dir)

                # get parameters for file number
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
                _all_gigatrons_params = read_gigatrons_from_zip(
                    get_zip_archive(
                        self.rootpath, self.year, self.month, self.day,
                        fileno, self.prefix
                    )
                )
                data_dict = _all_params | _all_frequencies | _all_gigatrons_params
                
                # add scanned parameters
                for j, p in enumerate(parameter):
                    assert p in data_dict
                    self.parameter[j,i] = data_dict[p]
                
                # number, use shape for image roi
                exposure_time = data_dict[self.exposure_time_param]*1e-5
                number_multiplier = self.photon/(
                    exposure_time*self.gamma*self.collection_solid_angle
                )

                # for now calculate N from whole image
                n1 = np.round(number_multiplier*np.sum(_image1))
                n2 = np.round(number_multiplier*np.sum(_image2))
                
                if n1 > 0:
                    self.number[i] = n2 / n1
                else:
                    self.number[i] = 0.0
                self.number1[i] = n1
                self.number2[i] = n2

            else:
                # subtract background images, take mean of images, then save
                # sometimes the triggering doesn't work and we only have one image
                # in that case just set the image to 0, so the outlier analysis will remove it
                if len(_raw_images) > 1:
                    _image = np.mean(
                        _raw_images[::2] - _raw_images[1::2],
                        axis=0
                    )
                else:
                    _image = np.zeros(np.shape(_raw_images[0]))
                self.save_image(_image, fileno, image_dir)

                # get parameters for file number
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
                _all_gigatrons_params = read_gigatrons_from_zip(
                    get_zip_archive(
                        self.rootpath, self.year, self.month, self.day,
                        fileno, self.prefix
                    )
                )
                data_dict = _all_params | _all_frequencies | _all_gigatrons_params
                
                # add scanned parameters
                for j, p in enumerate(parameter):
                    assert p in data_dict
                    self.parameter[j,i] = data_dict[p]

                
                # shape 
                try:
                    # actual horizontal and vertical!
                    h_fit, v_fit = calculate_cloud_size_from_image_1d_gaussian(
                            _image,
                            pixel_size=self.pixel_size,
                            bin_size=self.binning,
                            magnification=self.magnification
                        )
                    self.horizontal_width[i] = h_fit.width
                    self.vertical_width[i] = v_fit.width
                    self.horizontal_centre[i] = h_fit.centre
                    self.vertical_centre[i] = v_fit.centre
                except Exception as e: 
                    pass
                    # if the fit doesn't work then the size and position are 0 for now!
                    #print(
                    #    f"Error {e} occured in file {fileno} in size fit"
                    #)
                
                # number, use shape for image roi
                exposure_time = data_dict[self.exposure_time_param]*1e-5
                number_multiplier = self.photon/(
                    exposure_time*self.gamma*self.collection_solid_angle
                )
                
                #row_start = int(np.max([(self.vertical_centre[i]-2*self.vertical_width[i])*self.magnification/self.pixel_size/self.binning, 0]))
                #row_end = int(np.min([(self.vertical_centre[i]+2*self.vertical_width[i])*self.magnification/self.pixel_size/self.binning, 512/self.binning]))
                #col_start = int(np.max([(self.horizontal_centre[i]-2*self.horizontal_width[i])*self.magnification/self.pixel_size/self.binning, 0]))
                #col_end = int(np.min([(self.horizontal_centre[i]-2*self.horizontal_width[i])*self.magnification/self.pixel_size/self.binning, 512/self.binning]))
                
                # self.number[i] = number_multiplier*np.sum(_image[row_start:row_end, col_start:col_end])
                
                # for now calculate N from whole image
                self.number[i] = np.round(number_multiplier*np.sum(_image))
                
                # density
                v_horiz = self.horizontal_width[i]**2
                v_vert = self.vertical_width[i]
                try:
                    if v_horiz == 0 or v_vert == 0:
                        self.density[i] = 0.0
                    else:
                        self.density[i] = (3*self.number[i])/(4*np.pi*v_horiz*v_vert)
                except Exception as e: 
                    # if the fit doesn't work and the size and position are 0, it will try to devide by zero
                    print(
                        f"Error {e} occured in file {fileno} in density calculation"
                    )

        self.write_csv(data_dir)

        self.calculateMeanAndErr()
            
        return self
    
    def write_header(self) -> None:
        # write header into a txt (?) file, name with file numbers
        return
    
    def write_csv(self, dir: Path) -> None:
        self.df = pd.DataFrame(
            {
            "ImageNo": self.fileno, 
            "N": self.number, 
            "hWidth": self.horizontal_width,
            "vWidth": self.vertical_width,
            "hCtr": self.horizontal_centre,
            "vCtr": self.vertical_centre,
            "density": self.density
            }
        )
        for i, p in enumerate(self.parameter_names):
            self.df[p] = pd.Series(self.parameter[i], index=self.df.index)
        self.df.set_index('ImageNo', inplace=True)

        file_start_filled: str = str(self.file_start).zfill(3)
        file_stop_filled: str = str(self.file_stop).zfill(3)
        self.df.to_csv(dir / (self.create_file_prefix() + f"{file_start_filled}_to_{file_stop_filled}.csv"), index=True)

        return
    
    def create_dir(self) -> List[Path]:
        rootpath = Path(self.outputrootpath)
        day_filled: str = str(self.day).zfill(2)
        data_dir = rootpath.joinpath(
            "data",
            str(self.year),
            month_dict[self.month],
            day_filled
        )
        image_dir = rootpath.joinpath(
            "images",
            str(self.year),
            month_dict[self.month],
            day_filled
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        return [data_dir, image_dir]
    
    def create_file_prefix(self) -> str:
        day_filled: str = str(self.day).zfill(2)
        year_filled: str = str(self.year)[-2:]
        month_filled: str = month_dict[self.month][-3:]

        return f"{self.prefix}{day_filled}{month_filled}{year_filled}00_"
    
    def save_image(self, image, fileno, filedir) -> None:
        im = Image.fromarray(image)
        fileno_filled = str(fileno).zfill(3)
        im.save(filedir / (self.create_file_prefix() + f'{fileno_filled}.tif'))

    def remove_outliers(self, df: pd.DataFrame, parameters: List[str], z_threshold: float = 2.0):
        df['_idx'] = df.index
        df_list = df.groupby(parameters).agg(list)
        df_zscore = df_list[['N', 'hWidth', 'vWidth', 'hCtr', 'vCtr', 'density']].map(zscore)
        df_zscore['_idx'] = df_list['_idx']
        df_new = df.join(df_zscore.explode(['N', 'hWidth', 'vWidth', 'hCtr', 'vCtr', 'density', '_idx']).set_index('_idx'), rsuffix='_z')
        df_new.drop('_idx', axis=1, inplace=True)
        idx_names = df_new[((np.abs(df_new['N_z']) > z_threshold) | (np.abs(df_new['hCtr_z']) > z_threshold) | (np.abs(df_new['vCtr_z']) > z_threshold))].index
        df_final = df_new.drop(idx_names)
        return df_final

    def calculateMeanAndErr(self, removeOutliers: bool = True, z: float = 2.0):
        if removeOutliers:
            _df = self.remove_outliers(self.df, self.parameter_names, z_threshold=z)
        else:
            _df = self.df

        self.df_mean = _df.groupby(self.parameter_names).mean()
        self.df_err = _df.groupby(self.parameter_names).sem()

        return self
    
    def plot_2d(self, quantity, xlabel = '', ylabel = '', zlabel = '' , zerrlabel = 'err', xscale = 1.0, yscale = 1.0, zscale = 1.0, vmin=None, vmax=None):
        xx = self.df_mean.index.get_level_values(0).to_numpy() * xscale
        yy = self.df_mean.index.get_level_values(1).to_numpy() * yscale
        zz = self.df_mean[quantity].to_numpy() * zscale
        zz_err = self.df_err[quantity].to_numpy() * zscale

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

        if xlabel == '': xlabel = self.df_mean.index.names[0]
        if ylabel == '': ylabel = self.df_mean.index.names[1]

        pcm1 = ax1.pcolormesh(xx.reshape(len(set(xx)), len(set(yy))), yy.reshape(len(set(xx)), len(set(yy))), zz.reshape(len(set(xx)), len(set(yy))), vmin=vmin, vmax=vmax)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(zlabel)

        pcm2 = ax2.pcolormesh(xx.reshape(len(set(xx)), len(set(yy))), yy.reshape(len(set(xx)), len(set(yy))), zz_err.reshape(len(set(xx)), len(set(yy))))
        ax2.set_xlabel(xlabel)
        ax2.set_title(zerrlabel)

        f.colorbar(pcm1, ax=ax1)
        f.colorbar(pcm2, ax=ax2)

        return self
    
    def plot_1d(self, quantity, excluded_idx = [], xlabel = '', ylabel = '', xscale = 1.0, yscale = 1.0, xoffset=0.0):
        if xlabel == '': xlabel = self.df_mean.index.names[0]
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        xx = np.delete(self.df_mean.index.to_numpy(), excluded_idx) * xscale - xoffset
        yy = np.delete(self.df_mean[quantity].to_numpy(), excluded_idx) * yscale
        err = np.delete(self.df_err[quantity].to_numpy(), excluded_idx) * yscale
        ax.errorbar(xx, yy, yerr=err, fmt="ok")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return self
    
    def plot_1d_all(self, excluded_idx=[], xscale=1.0, xlabel='', xunit=''):
        fig, ax = plt.subplots(3, 2, figsize=(8, 9))
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

        xx = np.delete(self.df_mean.index.to_numpy(), excluded_idx) * xscale
        if xlabel == '': xlabel = self.df_mean.index.names[0]
        if xunit == '': xlabel_add = ''
        else: xlabel_add = f' [{xunit}]'
        
        qtys = ['N','density','hWidth','vWidth','hCtr','vCtr']
        yscales = [1.0, 1e-6, 1e3, 1e3, 1e3, 1e3]
        ylabels = ['Molecule number', 'Density [cm$^{-3}$]', 'Horizontal width [mm]', 'Vertical width [mm]', 'Horizontal center [mm]', 'Vertical center [mm]']
        for qty, yscale, axis, ylabel in zip(qtys, yscales, ax.flatten(), ylabels):
            yy = np.delete(self.df_mean[qty].to_numpy(), excluded_idx) * yscale
            err = np.delete(self.df_err[qty].to_numpy(), excluded_idx) * yscale
            axis.errorbar(xx, yy, yerr=err, fmt="ok")
            axis.set_xlabel(xlabel + xlabel_add)
            axis.set_ylabel(ylabel)

        return self
    
    def plot_temperature(self, excluded_idx_h = [], excluded_idx_v = []):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        fig.subplots_adjust(wspace=0.3)

        xh = (np.delete(self.df_mean.index.to_numpy(), excluded_idx_h) * 1e-5) ** 2
        yh = np.delete(self.df_mean['hWidth'].to_numpy(), excluded_idx_h) ** 2
        errh = (2 * np.delete(self.df_err['hWidth'].to_numpy(), excluded_idx_h) / np.delete(self.df_mean['hWidth'].to_numpy(), excluded_idx_h)) * yh
        fith = fit_linear(xh, yh, errh)
        temph = fith.slope * self.mass * cn.u / cn.k * 1e6
        temph_err = fith.slope_err * self.mass * cn.u / cn.k * 1e6
        ax1.errorbar(fith.x * 1e6, fith.y * 1e6, yerr=np.abs(errh)*1e6, fmt="ok")
        ax1.plot(fith.x_fine * 1e6, fith.y_fine * 1e6, '-r')
        ax1.set_xlabel('$t^2$ [ms$^2$]')
        ax1.set_ylabel('$\\sigma_h^2$ [mm$^2$]')
        ax1.set_title(f"T$_h$ = {temph:.2f}$\\pm${temph_err:.2f} uK")

        xv = (np.delete(self.df_mean.index.to_numpy(), excluded_idx_v) * 1e-5) ** 2
        yv = np.delete(self.df_mean['vWidth'].to_numpy(), excluded_idx_v) ** 2
        errv = (2 * np.delete(self.df_err['vWidth'].to_numpy(), excluded_idx_v) / np.delete(self.df_mean['vWidth'].to_numpy(), excluded_idx_v)) * yv
        fitv = fit_linear(xv, yv, errv)
        tempv = fitv.slope * self.mass * cn.u / cn.k * 1e6
        tempv_err = fitv.slope_err * self.mass * cn.u / cn.k * 1e6
        ax2.errorbar(fitv.x * 1e6, fitv.y * 1e6, yerr=np.abs(errv)*1e6, fmt="ok")
        ax2.plot(fitv.x_fine * 1e6, fitv.y_fine * 1e6, '-r')
        ax2.set_xlabel('$t^2$ [ms$^2$]')
        ax2.set_ylabel('$\\sigma_v^2$ [mm$^2$]')
        ax2.set_title(f"T$_v$ = {tempv:.2f}$\\pm${tempv_err:.2f} uK")

        return self
    
    def plot_exp_decay(self, quantity='N', excluded_idx=[], ylabel='Molecule number'):
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        x = np.delete(self.df_mean.index.to_numpy(), excluded_idx) * 1e-5
        y = np.delete(self.df_mean[quantity].to_numpy(), excluded_idx)
        err = np.delete(self.df_err[quantity].to_numpy(), excluded_idx)
        fit = fit_exponential_with_offset2(x, y, err)
        tau = fit.rate * 1e3
        tau_err = fit.rate_err * 1e3
        ax.errorbar(fit.x * 1e3, fit.y, yerr=err, fmt="ok")
        ax.plot(fit.x_fine * 1e3, fit.y_fine, '-r')
        ax.set_xlabel('$t$ [ms]')
        ax.set_ylabel(ylabel)
        ax.set_title(f"$\\tau$ = {tau:.2f}$\\pm${tau_err:.2f} ms")

        return self