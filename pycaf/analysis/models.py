from typing import Union, Tuple
from pydantic import BaseModel
import numpy as np


class Pattern(BaseModel):
    name: str
    time: np.ndarray
    event: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class Fit(BaseModel):
    func_str: str
    args_str: str
    x: np.ndarray = np.array([])
    y: np.ndarray = np.array([])
    err: Union[None, np.ndarray] = None
    x_fine: np.ndarray = np.array([])
    y_fine: np.ndarray = np.array([])

    class Config:
        arbitrary_types_allowed = True


class Fit2D(BaseModel):
    func: str
    x: np.ndarray = np.array([])
    y: np.ndarray = np.array([])
    data: np.ndarray = np.array([[]])

    class Config:
        arbitrary_types_allowed = True


class LinearFit(Fit):
    slope: float
    intercept: float


class GaussianFitWithoutOffset(Fit):
    amplitude: float
    centre: float
    width: float


class GaussianFitWithOffset(GaussianFitWithoutOffset):
    offset: float


class ExponentialFitWithoutOffset(Fit):
    amplitude: float
    centre: float
    rate: float


class ExponentialFitWithOffset(ExponentialFitWithoutOffset):
    offset: float


class GaussianFitWithoutOffset2D(Fit2D):
    amplitude: float
    centre: Tuple[float, float]
    width: Tuple[float, float]


class GaussianFitWithOffset2D(GaussianFitWithoutOffset2D):
    offset: float


class LorentzianFitWithoutOffset(Fit):
    amplitude: float
    centre: float
    width: float


class LorentzianFitWithOffset(LorentzianFitWithoutOffset):
    offset: float
