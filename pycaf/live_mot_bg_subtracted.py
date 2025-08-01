from typing import Union, List
import numpy as np
from PIL import Image
import os
import re
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

from pycaf.experiment import Experiment


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


class LiveMOTBGSubtracted(Experiment):
    def __init__(
        self,
        config_path: str,
        interval: Union[int, float],
        scripts: List[str],
        parameters: List[str],
        values: List[Union[int, float]],
        iteration: int = 5,
        color_max: int = 50
    ) -> None:
        super().__init__(config_path, interval)
        self.image_dirpath = self.config["temp_image_path"]
        self.scripts = scripts
        self.parameters = parameters
        self.values = values
        self.iteration = iteration
        self.color_max = color_max
        self.old_color_max = color_max
        self.image = None
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.paused = False
        self.ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            repeat=True
        )
        axprev = self.fig.add_axes([0.7, 0.030, 0.1, 0.04])
        axnext = self.fig.add_axes([0.81, 0.030, 0.1, 0.04])
        axstop = self.fig.add_axes([0.1, 0.030, 0.1, 0.04])
        self.bstop = Button(axstop, 'stop')
        self.bnext = Button(axnext, 'up')
        self.bprev = Button(axprev, 'down')

    def up(self, event):
        self.color_max = self.old_color_max + 10

    def down(self, event):
        self.color_max = self.old_color_max - 10

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
        self.ax.clear()
        self.ax.grid(False)
        self.old_color_max = self.color_max
        self.bnext.on_clicked(self.up)
        self.bprev.on_clicked(self.down)
        self.bstop.on_clicked(self.toggle_pause)
        for k in range(len(self.scripts)):
            self.scan(
                self.scripts[k],
                motmaster_parameters_with_values={
                    self.parameters[k]: [self.values[k]]
                },
                n_iter=1
            )
            try:
                images = self.read_images()
            except Exception as e:
                print(f"Exception {e} occured in during reading images")
            try:
                self.delete_images()
            except Exception as e:
                print(f"Exception {e} occured in during deleting images")
            if len(images) >= 2:
                _image = images[0::2, :, :] - images[1::2, :, :]
                self.image = np.sum(_image, axis=0)
            _img = self.ax.imshow(self.image)
            _img.set_clim(0, self.color_max)
            self.fig.canvas.draw()
            self.fig.suptitle(
                f"{self.scripts[k]}:{self.parameters[k]}->{self.values[k]} with cmax: {self.color_max}"
            )
            self.fig.canvas.flush_events()
            time.sleep(self.interval)
        return None
        

if __name__ == "__main__":
    config_path = "C:\\ControlPrograms\\pycaf\\config_bec.json"
    interval = 0.1
    scripts = ["AMOTBasic", "AMOTBasic"]
    parameters = ["yagONorOFF", "yagONorOFF"]
    values = [10.0, 1.0]
    iteration = 1
    live_mot = LiveMOTBGSubtracted(
        config_path=config_path,
        interval=interval,
        scripts=scripts,
        parameters=parameters,
        values=values,
        iteration=iteration,
        color_max=100
    )
    plt.show()
