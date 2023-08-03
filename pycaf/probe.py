from typing import Callable, Dict, List, Tuple, Any
from zipfile import ZipFile
import numpy as np
from pathlib import Path
from datetime import datetime

from .analysis import (
    create_file_list,
    read_parameters_from_zip,
    read_images_from_zip,
    read_digital_patterns_from_zip,
    read_analog_patterns_from_zip,
    read_time_of_flight_from_zip,
    calculate_molecule_number_from_fluorescent_images,
    calculate_optical_density_from_absorption_images,
    calculate_atom_number_from_absorption_images,
    calculate_cloud_size_from_image_1d_gaussian,
    calculate_cloud_size_from_image_2d_gaussian,
    crop_images
)
from .visualizations import (
    MatplotlibDisplay,
    PlotlyDisplay
)
from .analysis import (
    Pattern,
    Study,
    fit_gaussian_with_offset,
    fit_gaussian_without_offset
)

# FIXME: The methods are not correct


class Probe():
    def __init__(
        self,
        root: str,
        year: int,
        month: int,
        day: int,
        display_library: str = "plotly",
        *args,
        **kwargs
    ) -> None:
        month = str(month).zfill(2)
        month_name = datetime.strptime(month, "%m").strftime("%b")
        self.rootpath = str(
            Path(root).joinpath(
                f"{year}",
                f"{month}{month_name}",
                f"{day}",
                f"CaF{day}{month_name}{year-2000}00_"
            )
        )
        if display_library == "matplotlib":
            self.plot = MatplotlibDisplay()
        elif display_library == "plotly":
            self.display_lib = PlotlyDisplay()
        self.args = args
        self.kwargs = kwargs
        self.fl_strs = ["f", "F", "fluoresence", "Fluoresence"]
        self.abs_strs = ["a", "A", "absorption", "Absorption"]
        self.fit_func_strs = \
            {
                "fit_gaussian_with_offset": fit_gaussian_with_offset,
                "fit_gaussian_without_offset": fit_gaussian_without_offset,
                "fit_exponential_with_offset": 0,
                "fit_exponential_without_offset": 0
            }
        self._add_essential_parameters()
        self._set_attributes()

    def _add_essential_parameters(
        self
    ) -> None:
        self.pixel_size: float = None
        self.bin_size: int = None
        self.magnification: float = None
        self.crop: bool = None
        self.crop_centre: Tuple[int, int] = None
        self.crop_width: int = None
        self.crop_height: int = None
        self.image_height: int = None
        self.image_width: int = None
        self.full_well_capacity: float = None
        self.bits_per_channel: int = None
        self.exposure_time: float = None
        self.gamma: float = None
        self.collection_solid_angle: float = None
        self.eta_q: float = None

    def _set_attributes(
        self
    ) -> None:
        self.__dict__.update(self.kwargs)

    def __setattr__(
        self,
        key: Any,
        value: Any
    ) -> None:
        self.__dict__[key] = value

    def create_study(
        self,
        repetition_rate: int,
        n_trigger: int = 1,
        repetition_skip: int = 0,
        cloud: str = "CaF",
        detection_type: str = "fluorescent",
        file_no_start: int = None,
        file_no_stop: int = None,
        background_file_start: int = None,
        background_file_stop: int = None,
        files: List[int] = None,
        background_files: List[int] = None,
        exclude_files: List[int] = None,
    ) -> None:
        self.study = Study()
        self.numbers = None
        self.numbers_err = None
        self.horizontal_size = None
        self.horizontal_size_err = None
        self.vertical_size = None
        self.vertical_size_err = None
        if not files:
            files, background_files = create_file_list(
                file_no_start,
                file_no_stop,
                background_file_start,
                background_file_stop,
                exclude_files
            )
        n_files = len(files)
        assert n_files == len(background_files)
        self.study.detection_type = detection_type
        self.study.cloud = cloud
        self.study.n_trigger = n_trigger
        self.study.repetition_rate = repetition_rate
        self.study.repetition_skip = repetition_skip
        self.study.files = np.array(files, dtype=int)
        self.study.background_files = np.array(background_files, dtype=int)
        self.study.parameters = np.empty((n_files), dtype=float)
        data_len = repetition_rate-repetition_skip
        if detection_type in self.fl_strs:
            self.study.images = np.empty(
                (n_files, data_len, n_trigger,
                    self.crop_height, self.crop_width),
                dtype=float
            )
            self.study.numbers = np.empty(
                (n_files, data_len, n_trigger),
                dtype=float
            )
            self.study.horizontal_fits = np.empty(
                (n_files, data_len, n_trigger, 4),
                dtype=float
            )
            self.study.vertical_fits = np.empty(
                (n_files, data_len, n_trigger, 4),
                dtype=float
            )
        if detection_type in self.abs_strs:
            self.study.images = np.empty(
                (n_files, data_len, n_trigger-2,
                    self.crop_height, self.crop_width),
                dtype=float
            )
            self.study.numbers = np.empty(
                (n_files, data_len, n_trigger-2),
                dtype=float
            )
            self.study.horizontal_fits = np.empty(
                (n_files, data_len, n_trigger-2, 4),
                dtype=float
            )
            self.study.vertical_fits = np.empty(
                (n_files, data_len, n_trigger-2, 4),
                dtype=float
            )
        return self

    def _get_images_processed_for_fluoresence(
        self,
        images: np.ndarray,
        background_file: int,
    ) -> np.ndarray:
        if background_file:
            bg_file = str(background_file).zfill(3)
            bg_archive = ZipFile(f"{self.rootpath}_{bg_file}")
            _bg_images = np.array(read_images_from_zip(bg_archive))
            images -= np.mean(_bg_images, axis=0)
        if self.crop and self.crop_centre:
            images = crop_images(
                images=images,
                centre=self.crop_centre,
                height=self.crop_height,
                width=self.crop_width
            )
        return images

    def _get_images_processed_for_absorption(
        self,
        images: np.ndarray
    ) -> np.ndarray:
        images = optical_density_from_absorption_images(
            images[0::3],
            images[1::3],
            images[2::3]
        )
        if self.crop and self.crop_centre:
            images = crop_images(
                images=images,
                centre=self.crop_centre,
                height=self.crop_height,
                width=self.crop_width
            )
        return images

    def _post_process(
        self
    ) -> None:
        self.x = self.study.parameters
        data_len = self.study.repetition_rate-self.study.repetition_skip
        if self.study.detection_type in self.fl_strs:
            if self.study.type == "number":
                var = self.study.numbers
                if self.study.n_trigger == 1:
                    self.numbers = np.mean(var, axis=1)
                    self.numbers_err = np.std(var, axis=1)/np.sqrt(data_len)
                elif self.study.n_trigger == 2:
                    self.numbers = np.mean(var[:, 1::2]/var[:, 0::2], axis=1)
                    self.numbers_err = np.std(var[:, 1::2]/var[:, 0::2], axis=1)\
                        / np.sqrt(data_len)
            elif self.study.type == "size":
                h_size = self.study.horizontal_fits[:, :, 2]
                v_size = self.study.vertical_fits[:, :, 2]
                if self.study.n_trigger == 1:
                    self.horizontal_size = np.mean(h_size, axis=1)
                    self.horizontal_size_err = np.std(h_size, axis=1)\
                        / np.sqrt(data_len)
                    self.vertical_size = np.mean(v_size, axis=1)
                    self.vertical_size_err = np.std(v_size, axis=1)\
                        / np.sqrt(data_len)
        elif self.study.detection_type in self.abs_strs:
            if self.study.type == "number":
                self.y = self.study.numbers
                self.yerr = 0
        return None

    def number(
        self,
        parameter: str,
    ) -> None:
        for file_no, background_file in zip(
            self.study.files,
            self.study.background_files
        ):
            index = file_no-np.min(self.study.files)
            archive = ZipFile(f"{self.rootpath}_{str(file_no).zfill(3)}")
            parameters = read_parameters_from_zip(archive)
            assert parameter in list(parameters.keys())
            _images = np.array(read_images_from_zip(archive))
            if self.study.detection_type in self.fl_strs:
                assert self.full_well_capacity is not None
                assert self.bits_per_channel is not None
                assert self.collection_solid_angle is not None
                assert self.eta_q is not None
                assert self.exposure_time is not None
                assert self.gamma is not None
                _images = self._get_images_processed_for_fluoresence(
                    images=_images,
                    background_file=background_file
                )
                numbers = cloud_number_from_fluorescent_images(
                    images=_images,
                    full_well_capacity=self.full_well_capacity,
                    bits_per_channel=self.bits_per_channel,
                    exposure_time=self.exposure_time,
                    gamma=self.gamma,
                    collection_solid_angle=self.collection_solid_angle,
                    eta_q=self.eta_q
                )
            elif self.study.detection_type in self.abs_strs:
                assert np.mod(np.shape(_images)[0], 3) == 0
                assert self.pixel_size is not None
                assert self.bin_size is not None
                assert self.magnification is not None
                _images = self._get_images_processed_for_absorption(
                    images=_images
                )
                numbers = cloud_number_from_absorption_images(
                    images=_images,
                    pixel_size=self.pixel_size,
                    bin_size=self.bin_size,
                    magnification=self.magnification,
                    saturation=0
                )
            self.study.parameters[index] = parameters[parameter]
            self.study.files[index] = file_no
            self.study.numbers[index, :, :] = numbers
            self.study.images[index, :, :, :, :] = _images
        self._post_process()
        return self

    def size(
        self,
        parameter: str,
        fitting_type: str = "1d_gaussian"
    ) -> None:
        assert self.pixel_size is not None
        assert self.bin_size is not None
        assert self.magnification is not None
        if fitting_type == "1d_gaussian":
            cloud_size_from_images: Callable = \
                cloud_sizes_from_images_using_1d_gaussian
        elif fitting_type == "2d_gaussian":
            cloud_size_from_images: Callable = \
                cloud_sizes_from_images_using_2d_gaussian
        for file_no, background_file in zip(
            self.study.files,
            self.study.background_files
        ):
            index = file_no-np.min(self.study.files)
            archive = ZipFile(f"{self.rootpath}_{str(file_no).zfill(3)}")
            parameters = read_parameters_from_zip(archive)
            assert parameter in list(parameters.keys())
            _images = read_images_from_zip(archive)
            if self.study.detection_type in self.fl_strs:
                _images = self._get_images_processed_for_fluoresence(
                    images=_images,
                    background_file=background_file
                )
            elif self.study.detection_type in self.abs_strs:
                assert np.mod(np.shape(_images)[0], 3) == 0
                _images = self._get_images_processed_for_absorption(
                    images=_images
                )
            vertical_fits, horizontal_fits, fit_info = \
                cloud_size_from_images(
                    image=_images,
                    pixel_size=self.pixel_size,
                    bin_size=self.bin_size,
                    magnification=self.magnification
                )
            self.study.horizontal_fits[index, :, :, :] = horizontal_fits
            self.study.vertical_fits[index, :, :, :] = vertical_fits
            self.study.parameters[index] = parameters[parameter]
            self.study.files[index] = file_no
            self.study.images[index, :, :, :, :] = _images
        self.study.info.update({"size_fit": fit_info})
        self._post_process()
        return self

    def compare_patterns(
        self,
        files: List[int] = None,
        parameters: List[str] = None,
    ) -> None:
        for file_no in files:
            archive = ZipFile(f"{self.rootpath}_{str(file_no).zfill(3)}")
            digital_patterns: Dict[str, Pattern] = \
                read_digital_patterns_from_zip(archive)
            if not parameters:
                _patterns = {}
                for parameter in parameters:
                    if parameter in digital_patterns:
                        _patterns[parameter] = digital_patterns[parameter]
            else:
                _patterns = digital_patterns
            self.study.digital_patterns[file_no] = _patterns
            analog_patterns: Dict[str, Pattern] = \
                read_analog_patterns_from_zip(archive)
            if not parameters:
                _patterns = {}
                for parameter in parameters:
                    if parameter in analog_patterns:
                        _patterns[parameter] = analog_patterns[parameter]
            else:
                _patterns = analog_patterns
            self.study.analog_patterns[file_no] = _patterns
        return self

    def time_of_flight(
        self,
        parameter: str = None
    ) -> None:
        return self

    def fit(
        self,
        function_type: str,
        n_fine: int = 100
    ) -> None:
        func: Callable = self.fit_func_strs[function_type]
        self.study.fit = func(
            self.x,
            self.y,
            self.yerr,
            n_fine
        )
        return self

    def display(
        self,
        **kwargs
    ) -> Tuple[Any, Any]:
        if self.study.type:
            fig, ax = self.display_lib(self.study, **kwargs)
            return fig, ax
        return None
