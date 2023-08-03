from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from ..analysis import (
    Study
)
from .display import Display


class MatplotlibDisplay(Display):
    def __init__(
        self
    ) -> None:
        pass

    def __call__(
        self,
        study: Study,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        if "xscale" in kwargs:
            x /= kwargs["xscale"]
        if "yscale" in kwargs:
            y /= kwargs["yscale"]
        figsize = kwargs["figsize"] if "figsize" in kwargs else (10, 6)
        label = kwargs["label"] if "label" in kwargs else "y"
        fmt = kwargs["fmt"] if "fmt" in kwargs else "ok"
        fit_fmt = kwargs["fit_fmt"] if "fit_fmt" in kwargs else "-r"
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if yerr:
            ax.errorbar(x, y, yerr=yerr, fmt=fmt, label=label)
        else:
            ax.plot(x, y, fmt=fmt, label=label)
        if study.fit:
            ax.plot(study.fit.x_fine, study.fit.y_fine, fmt=fit_fmt)
        if "xlim" in kwargs:
            ax.set_xlim(kwargs["xlim"])
        if "ylim" in kwargs:
            ax.set_ylim(kwargs["ylim"])
        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"])
        if "title" in kwargs:
            ax.set_title(kwargs["title"])
        ax.legend()
        return fig, ax
