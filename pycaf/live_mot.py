from typing import Union, Tuple
import numpy as np
from PIL import Image
import os

from .experiment import Experiment
from .analysis import (
    fit_gaussian_with_offset,
    GaussianFitWithOffset,
    fit_exponential_with_offset,
    ExponentialFitWithOffset
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _get_image_from_file(
    filepath: str
) -> np.ndarray:
    # BUG FIX: Python was not realeasing the file resource for deletion
    with Image.open(filepath, 'r') as imagefile:
        image = np.array(imagefile, dtype=float)
    return image


class LiveMOT(Experiment):
    def __init__(
        self,
        config_path: str,
        interval: Union[int, float],
        script: str,
        field_parameter: str,
        field_parameter_on_value: float,
        field_parameter_off_value: float,
        is_lifetime_required: bool = False,
        timegap_in_ms: int = 1000
    ) -> None:
        super().__init__(config_path, interval)
        self.image_dirpath = self.config["temp_image_path"]
        self.script = script
        self.field_parameter = field_parameter
        self.field_parameter_on_value = field_parameter_on_value
        self.field_parameter_off_value = field_parameter_off_value
        self.is_lifetime_required = is_lifetime_required
        self.timegap_in_ms = timegap_in_ms
        super().connect()

    def lifetime_analysis(
        self,
        images: np.ndarray,
        timegap_in_ms: int
    ) -> ExponentialFitWithOffset:
        numbers = np.sum(images, axis=(1, 2))
        timesteps = timegap_in_ms*np.arange(0, len(numbers))
        lifetime_fit = fit_exponential_with_offset(numbers, timesteps)
        return lifetime_fit

    def number_analysis(
        self,
        image: np.ndarray
    ) -> float:
        return np.sum(image)

    def size_analysis(
        self,
        image: np.ndarray
    ) -> Tuple[GaussianFitWithOffset, GaussianFitWithOffset]:
        h_profile = np.sum(image, axis=0)
        v_profile = np.sum(image, axis=1)
        h_profile_fit = fit_gaussian_with_offset(
            h_profile,
            np.arange(0, len(h_profile))
        )
        v_profile_fit = fit_gaussian_with_offset(
            v_profile,
            np.arange(0, len(v_profile))
        )
        return (h_profile_fit, v_profile_fit)

    def read_images(
        self
    ) -> np.ndarray:
        images = []
        for filename in os.listdir(self.image_dirpath):
            if '.tif' in filename:
                images.append(_get_image_from_file(filename))
        return np.array(images)

    def delete_images(
        self
    ) -> None:
        for filename in os.listdir(self.image_dirpath):
            if '.tif' in filename:
                os.remove(os.path.join(self.image_dirpath, filename))
        return None

    def __call__(
        self,
    ) -> Tuple[
            float,
            ExponentialFitWithOffset,
            GaussianFitWithOffset,
            GaussianFitWithOffset
    ]:
        lifetime_fit, number = None, None
        h_profile_fit, v_profile_fit = None, None
        self.motmaster_single_run(
            self.script,
            self.field_parameter,
            self.field_parameter_on_value
        )
        images = self.read_images()
        self.delete_images()
        self.motmaster_single_run(
            self.script,
            self.field_parameter,
            self.field_parameter_off_value
        )
        bg = self.read_images()
        self.delete_images()
        images -= bg
        if self.is_lifetime_required:
            lifetime_fit = self.lifetime_analysis(
                images,
                self.timegap_in_ms
            )
        else:
            image = np.mean(images, axis=0)
            number = self.number_analysis(image)
            h_profile_fit, v_profile_fit = self.size_analysis(image)
        return number, lifetime_fit, h_profile_fit, v_profile_fit


if __name__ == "__main__":
    config_path = "C:\\ControlPrograms\\pycaf\\config_bec.json"
    interval = 0.1
    script = "MOTBasicMultiTrigger"
    field_parameter = "MOTCoilsOnValue"
    field_parameter_on_value = 1.0
    field_parameter_off_value = 0.0
    is_lifetime_required = False
    timegap_in_ms = 25
    live_mot = LiveMOT(
        config_path=config_path,
        interval=interval,
        script=script,
        field_parameter=field_parameter,
        field_parameter_on_value=field_parameter_on_value,
        field_parameter_off_value=field_parameter_off_value,
        is_lifetime_required=is_lifetime_required,
        timegap_in_ms=timegap_in_ms
    )
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    line, = ax.plot([], [], lw=2)
    tdata, ydata = [], []

    def iterator(
        data: Tuple[
            float,
            ExponentialFitWithOffset,
            GaussianFitWithOffset,
            GaussianFitWithOffset
        ]
    ):
        _, lifetime_fit, _, _ = data
        ax.clear()
        line.set_data(lifetime_fit.x_fine, lifetime_fit.y_fine)
        return line

    ani = animation.FuncAnimation(
        fig,
        iterator,
        live_mot,
        blit=True,
        interval=5,
        repeat=False
    )
    plt.show()
