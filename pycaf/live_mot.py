from typing import Union, Tuple
import numpy as np
from PIL import Image
import os
import re
import time

from pycaf.experiment import Experiment
from pycaf.analysis import (
    fit_gaussian_without_offset,
    GaussianFitWithoutOffset,
    fit_exponential_without_offset,
    ExponentialFitWithoutOffset
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
        timegap_in_ms: int = 1000,
        crop_row_start: int = 0,
        crop_row_end: int = -1,
        crop_col_start: int = 0,
        crop_col_end: int = -1
    ) -> None:
        super().__init__(config_path, interval)
        super().connect()
        self.image_dirpath = self.config["temp_image_path"]
        self.script = script
        self.timegap_in_ms = timegap_in_ms
        self.crop_row_start = crop_row_start
        self.crop_row_end = crop_row_end
        self.crop_col_start = crop_col_start
        self.crop_col_end = crop_col_end
        self.lifetime_fit, self.h_profile_fit, self.v_profile_fit = \
            None, None, None
        self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 10))
        self.fig.subplots_adjust(hspace=0.2, wspace=0.02)
        self.paused = False
        self.number_list = np.array([], dtype=float)
        self.lifetime_list = np.array([], dtype=float)
        self.ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            repeat=True
        )
        self.fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(
        self,
        *args,
        **kwargs
    ) -> None:
        if self.paused:
            self.ani.resume()
        else:
            self.ani.pause()
        self.paused = not self.paused

    def lifetime_analysis(
        self,
        images: np.ndarray,
        timegap_in_ms: int
    ) -> ExponentialFitWithoutOffset:
        numbers = np.sum(images, axis=(1, 2))
        timesteps = timegap_in_ms*np.arange(0, len(numbers))
        lifetime_fit = fit_exponential_without_offset(timesteps, numbers)
        return lifetime_fit

    def number_analysis(
        self,
        image: np.ndarray
    ) -> float:
        return np.sum(image)

    def size_analysis(
        self,
        images: np.ndarray
    ) -> Tuple[GaussianFitWithoutOffset, GaussianFitWithoutOffset]:
        image = images[0, :, :]
        h_profile = np.sum(image, axis=0)
        v_profile = np.sum(image, axis=1)
        h_profile_fit = fit_gaussian_without_offset(
            np.arange(0, len(h_profile)),
            h_profile
        )
        v_profile_fit = fit_gaussian_without_offset(
            np.arange(0, len(v_profile)),
            v_profile,
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

    def animate(
        self,
        frame: int
    ) -> None:
        self._call_motmaster_and_analyse_image()
        if self.lifetime_fit is not None:
            for iax in self.ax.flatten():
                iax.clear()
            self.ax[0, 0].plot(
                self.lifetime_fit.x,
                self.lifetime_fit.y,
                "ok"
            )
            self.ax[0, 0].plot(
                self.lifetime_fit.x_fine,
                self.lifetime_fit.y_fine,
                "-r",
                label=f"Lifetime: {self.lifetime_fit.rate:.2f} ms"
            )
        if self.h_profile_fit is not None:
            self.ax[1, 0].plot(
                self.h_profile_fit.x,
                self.h_profile_fit.y,
                "or",
                label="Horizontal"
            )
            self.ax[1, 0].plot(
                self.h_profile_fit.x_fine,
                self.h_profile_fit.y_fine,
                "-r",
                label=f"Horiz. width: {self.h_profile_fit.width:.2f}"
            )
        if self.v_profile_fit is not None:
            self.ax[1, 0].plot(
                self.v_profile_fit.x,
                self.v_profile_fit.y,
                "ob",
                label="Vertical"
            )
            self.ax[1, 0].plot(
                self.v_profile_fit.x_fine,
                self.v_profile_fit.y_fine,
                "-b",
                label=f"Vert. width: {self.v_profile_fit.width:.2f}"
            )
        if len(self.number_list):
            self.ax[0, 1].plot(
                np.arange(0, len(self.number_list)),
                self.number_list,
                "-ok",
                label="Total Count"
            )
        if len(self.lifetime_list):
            self.ax[1, 1].plot(
                np.arange(0, len(self.lifetime_list)),
                self.lifetime_list,
                "-ok",
                label="Lifetime"
            )
            self.ax[0, 1].yaxis.set_label_position("right")
            self.ax[0, 1].yaxis.tick_right()
            self.ax[1, 1].yaxis.set_label_position("right")
            self.ax[1, 1].yaxis.tick_right()
            self.ax[0, 1].set_xlim((0, len(self.number_list)+10))
            self.ax[1, 1].set_xlim((0, len(self.lifetime_list)+10))
            self.ax[0, 0].set_xlabel("time in ms")
            self.ax[1, 1].set_xlabel("Iteration")
            self.ax[0, 1].set_xlabel("Iteration")
            self.ax[1, 0].set_xlabel("Distance [a. u.]")
            for iax in self.ax.flatten():
                iax.legend()
            self.fig.canvas.draw()
            self.fig.suptitle(f"Current frame no. : {frame}")
            self.fig.canvas.flush_events()
            time.sleep(self.interval)
        return None

    def _call_motmaster_and_analyse_image(
        self
    ) -> None:
        lifetime_fit, h_profile_fit, v_profile_fit = None, None, None
        self.motmaster_single_run(self.script)
        try:
            images = self.read_images()
        except Exception as e:
            print(f"Exception {e} occured in during reading images")
        try:
            self.delete_images()
        except Exception as e:
            print(f"Exception {e} occured in during deleting images")
        if len(images) > 1:
            bg = images[-1, :, :]
            images -= bg
            images = images[
                1:,
                self.crop_row_start: self.crop_row_end,
                self.crop_col_start: self.crop_col_end
            ]
        try:
            lifetime_fit = self.lifetime_analysis(
                images,
                self.timegap_in_ms
            )
            h_profile_fit, v_profile_fit = self.size_analysis(images)
        except Exception as e:
            print(f"Error {e} occured in fitting.")
        time.sleep(0.1)
        self.lifetime_fit = lifetime_fit
        self.h_profile_fit = h_profile_fit
        self.v_profile_fit = v_profile_fit
        self.lifetime_list = np.append(self.lifetime_list, lifetime_fit.rate)
        self.number_list = np.append(self.number_list, lifetime_fit.y[0])
        return None


if __name__ == "__main__":
    config_path = "C:\\ControlPrograms\\pycaf\\config.json"
    interval = 0.1
    script = "AMOTBasicLifetime"
    timegap_in_ms = 20
    live_mot = LiveMOT(
        config_path=config_path,
        interval=interval,
        script=script,
        timegap_in_ms=timegap_in_ms,
        crop_row_start=0,
        crop_row_end=-1,
        crop_col_start=0,
        crop_col_end=-1
    )
    plt.show()
