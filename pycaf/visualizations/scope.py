from typing import List, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..analysis import (
    get_zip_archive,
    read_images_from_zip
)


class Scope():
    def __init__(
        self,
        rootpath: str,
        year: int,
        month: int,
        day: int,
        prefix: str
    ) -> None:
        self.rootpath = rootpath
        self.year = year
        self.month = month
        self.day = day
        self.prefix = prefix
        self.clear()

    def clear(
        self
    ):
        self.row_start = 0
        self.row_stop = -1
        self.col_start = 0
        self.col_stop = -1
        self.start = 0
        self.stop = 0
        self.skip = 1
        self.style = "image"
        self.processed_images = np.array([], dtype=float)
        return self

    def range_with_image(
        self,
        start: int,
        stop: int,
        skip: List[int] = None
    ):
        self.start = start
        self.stop = stop
        self.skip = skip
        self.style = "image"
        return self

    def range_with_tofs(
        self,
        start: int,
        stop: int,
        skip: List[int] = None
    ):
        self.start = start
        self.stop = stop
        self.skip = skip
        self.style = "tof"
        return self

    def set_roi(
        self,
        row_start: int,
        row_stop: int,
        col_start: int,
        col_stop: int
    ):
        self.row_start = row_start
        self.row_stop = row_stop
        self.col_start = col_start
        self.col_stop = col_stop
        return self

    def _detect_image_shapes(
        self,
        fileno: int
    ) -> Tuple[int]:
        archive = get_zip_archive(
            self.rootpath,
            self.year,
            self.month,
            self.day,
            fileno,
            self.prefix
        )
        images = read_images_from_zip(archive)
        archive.close()
        return np.shape(images)

    def background_from_second_trigger(
        self
    ):
        image_shape = self._detect_image_shapes(self.start)
        processed_images = np.empty(image_shape, dtype=float)
        self.image_shape = image_shape
        self.scans = 0
        for fileno in range(self.start, self.stop+1, self.skip):
            archive = get_zip_archive(
                self.rootpath,
                self.year,
                self.month,
                self.day,
                fileno,
                self.prefix
            )
            all_images = read_images_from_zip(archive)
            archive.close()
            images, bg = all_images[0::2], all_images[1::2]
            images -= bg
            processed_images = np.append(images, axis=0)
            self.scans += 1
        self.processed_images = processed_images
        self.iterations = int(image_shape[0]/2)
        return self

    def background_from_second_file(
        self
    ):
        image_shape = self._detect_image_shapes(self.start)
        processed_images = np.empty(image_shape, dtype=float)
        self.image_shape = image_shape
        self.scans = 0
        for fileno in range(self.start, self.stop+1, 2):
            archive = get_zip_archive(
                self.rootpath,
                self.year,
                self.month,
                self.day,
                fileno,
                self.prefix
            )
            images = read_images_from_zip(archive)
            archive.close()
            archive = get_zip_archive(
                self.rootpath,
                self.year,
                self.month,
                self.day,
                fileno+1,
                self.prefix
            )
            bg = read_images_from_zip(archive)
            archive.close()
            images -= bg
            processed_images = np.append(images, axis=0)
            self.scans += 1
        self.processed_images = processed_images
        self.iterations = image_shape[0]
        return self

    def background_from_single_file(
        self,
        background_fileno: int
    ):
        archive = archive = get_zip_archive(
            self.rootpath,
            self.year,
            self.month,
            self.day,
            background_fileno,
            self.prefix
        )
        bg = read_images_from_zip(archive)
        archive.close()
        processed_images = np.empty(np.shape(bg), dtype=float)
        self.image_shape = np.shape(bg)
        self.scans = 0
        for fileno in range(self.start, self.stop+1, self.skip):
            archive = get_zip_archive(
                self.rootpath,
                self.year,
                self.month,
                self.day,
                fileno,
                self.prefix
            )
            images = read_images_from_zip(archive)
            archive.close()
            images -= bg
            processed_images = np.append(images, axis=0)
            self.scans += 1
        self.processed_images = processed_images
        self.iterations = np.shape(bg)[0]
        return self

    def calculate_n_in_roi(
        self
    ):
        images_roi = self.processed_images[
            :,
            self.row_start: self.row_stop,
            self.col_start: self.col_stop
        ]
        _n = np.sum(images_roi, axis=(1, 2)).reshape(
            (self.iterations, self.scans)
        )
        self.n = np.mean(_n, axis=0)
        self.dn = np.std(_n, axis=0)/(2*np.sqrt(len(_n)))
        return self

    def display_variation(
        self,
        xparam: List[Union[int, float]],
        xlabel: str = "set xlabel",
        ylabel: str = "set ylabel",
        title: str = "set title",
        figsize: Tuple = (10, 6),
        fmt: str = "-ok"
    ):
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.errorbar(xparam, self.n, yerr=self.dn, fmt=fmt)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return self

    def display_mean_images(
        self,
        clim_start: Union[int, float] = 0,
        clim_stop: Union[int, float] = 100,
        title: str = "set title",
        xlabel: str = "set xlabel",
        ylabel: str = "set ylable",
        lw: Union[int, float] = 2,
        ec: str = "w",
        figsize: Tuple[int] = (8, 5)
    ):
        for k in range(self.scans):
            split_images = self.processed_images[
                k*self.iterations: (k+1)*self.iterations, :, :
            ]
            mean_images = np.mean(split_images, axis=0)
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            im = ax.imshow(mean_images, axis=0)
            ax.add_patch(
                patches.Rectangle(
                    (self.row_start, self.col_start),
                    self.col_stop-self.col_start,
                    self.row_stop-self.row_start,
                    lw=lw, ec=ec, fc='none'
                )
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(False)
            fig.colorbar(im, ax=ax)
            im.set_clim((clim_start, clim_stop))
        return self

    def display_all_images(
        self,
        clim_start: Union[int, float] = 0,
        clim_stop: Union[int, float] = 100,
        title: str = "set title",
        xlabel: str = "set xlabel",
        ylabel: str = "set ylable",
        lw: Union[int, float] = 2,
        ec: str = "w",
        figsize: Tuple[int] = (8, 5)
    ):
        for k in range(self.scans):
            split_images = self.processed_images[
                k*self.iterations: (k+1)*self.iterations, :, :
            ]
            fig, ax = plt.subplots(
                1,
                self.iterations,
                figsize=figsize,
                sharex=True,
                sharey=True
            )
            for i in range(self.iterations):
                im = ax[i].imshow(split_images[i, :, :], axis=0)
                ax[i].add_patch(
                    patches.Rectangle(
                        (self.row_start, self.col_start),
                        self.col_stop-self.col_start,
                        self.row_stop-self.row_start,
                        lw=lw, ec=ec, fc='none'
                    )
                )
                ax[i].grid(False)
                im.set_clim((clim_start, clim_stop))
            fig.colorbar(im, ax=ax.ravel())
        return self
