from typing import Tuple

from ..analysis import (
    Study
)
from .display import Display


class PlotlyDisplay(Display):
    def __init__(
        self
    ) -> None:
        pass

    def numbers_with_errorbars(
        self,
        study: Study,
        fmt: str = "ok",
        figsize: Tuple[int | float, int | float] = (10, 6),
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        xlabel: str = None,
        ylabel: str = None,
        xscale: float = None,
        yscale: float = None,
        title: str = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return fig, ax

    def sizes_with_errorbars(
        self,
        study: Study,
        fmt: str = "ok",
        figsize: Tuple[int | float, int | float] = (10, 6),
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        xlabel: str = None,
        ylabel: str = None,
        xscale: float = None,
        yscale: float = None,
        title: str = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return fig, ax

    def lifetime(
        self,
        study: Study,
        fmt: str = "ok",
        figsize: Tuple[int | float, int | float] = (10, 6),
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        xlabel: str = None,
        ylabel: str = None,
        xscale: float = None,
        yscale: float = None,
        title: str = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return fig, ax

    def temperature(
        self,
        study: Study,
        fmt: str = "ok",
        figsize: Tuple[int | float, int | float] = (10, 6),
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        xlabel: str = None,
        ylabel: str = None,
        xscale: float = None,
        yscale: float = None,
        title: str = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return fig, ax

    def patterns(
        self,
        study: Study,
        fmt: str = "ok",
        figsize: Tuple[int | float, int | float] = (10, 6),
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        xlabel: str = None,
        ylabel: str = None,
        xscale: float = None,
        yscale: float = None,
        title: str = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return fig, ax
