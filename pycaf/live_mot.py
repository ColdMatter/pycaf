from typing import Union, Tuple
import numpy as np
from PIL import Image
import os
import re
import time

from pycaf.experiment import Experiment
from pycaf.analysis import (
    fit_gaussian_with_offset,
    GaussianFitWithOffset,
    fit_exponential_with_offset,
    ExponentialFitWithOffset
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


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
        timegap_in_ms: int = 1000,
        crop_row_start: int = 0,
        crop_row_end: int = -1,
        crop_col_start: int = 0,
        crop_col_end: int = -1
    ) -> None:
        super().__init__(config_path, interval)
        self.image_dirpath = self.config["temp_image_path"]
        self.script = script
        self.field_parameter = field_parameter
        self.field_parameter_on_value = field_parameter_on_value
        self.field_parameter_off_value = field_parameter_off_value
        self.is_lifetime_required = is_lifetime_required
        self.timegap_in_ms = timegap_in_ms
        self.crop_row_start = crop_row_start
        self.crop_row_end = crop_row_end
        self.crop_col_start = crop_col_start
        self.crop_col_end = crop_col_end
        super().connect()

    def lifetime_analysis(
        self,
        images: np.ndarray,
        timegap_in_ms: int
    ) -> ExponentialFitWithOffset:
        numbers = np.sum(images, axis=(1, 2))
        timesteps = timegap_in_ms*np.arange(0, len(numbers))
        lifetime_fit = fit_exponential_with_offset(timesteps, numbers)
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
        filenames = list(os.listdir(self.image_dirpath))
        filenames.sort(key=natural_keys)
        for filename in filenames:
            if '.tif' in filename:
                imagepath = os.path.join(self.image_dirpath, filename)
                image = _get_image_from_file(imagepath)
                images.append(image)
        return np.array(images)

    def delete_images(
        self
    ) -> None:
        for filename in os.listdir(self.image_dirpath):
            if '.tif' in filename:
                os.remove(os.path.join(self.image_dirpath, filename))
        return None

    def __call__(
        self
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
        # self.delete_images()
        if self.is_lifetime_required:
            bg = images[-1, :, :]
            images -= bg
            images = images[
                :,
                self.crop_row_start: self.crop_row_end,
                self.crop_col_start: self.crop_col_end
            ]
            lifetime_fit = self.lifetime_analysis(
                images,
                self.timegap_in_ms
            )
        else:
            image = np.mean(images, axis=0)
            number = self.number_analysis(image)
            # h_profile_fit, v_profile_fit = self.size_analysis(image)
        time.sleep(0.1)
        return number, lifetime_fit, h_profile_fit, v_profile_fit


if __name__ == "__main__":
    config_path = "C:\\ControlPrograms\\pycaf\\config_bec.json"
    interval = 0.1
    script = "MOTBasicMultiTrigger"
    field_parameter = "MOTCoilsOnValue"
    field_parameter_on_value = 1.0
    field_parameter_off_value = 0.0
    is_lifetime_required = True
    timegap_in_ms = 25
    live_mot = LiveMOT(
        config_path=config_path,
        interval=interval,
        script=script,
        field_parameter=field_parameter,
        field_parameter_on_value=field_parameter_on_value,
        field_parameter_off_value=field_parameter_off_value,
        is_lifetime_required=is_lifetime_required,
        timegap_in_ms=timegap_in_ms,
        crop_row_start=0,
        crop_row_end=30,
        crop_col_start=10,
        crop_col_end=40
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    def animate(i):
        print(f"animated: {i}")
        _, lifetime_fit, _, _ = live_mot()
        ax.clear()
        ne = np.random.random((len(lifetime_fit.x), 2))
        ax.plot(
            lifetime_fit.x_fine,
            lifetime_fit.y_fine,
            "-r"
        )
        ax.plot(
            lifetime_fit.x+10*ne[:, 0],
            lifetime_fit.y+10*ne[:, 0],
            "ok"
        )
        ax.set_title(
            f"Current MOT lifetime: {lifetime_fit.rate+10*ne[0, 0]} ms"
        )
        # line.set_data(lifetime_fit.x, lifetime_fit.y)
        # fig.canvas.draw()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.5)

    ani = animation.FuncAnimation(fig, animate, interval=10, repeat=True)
    plt.show()
