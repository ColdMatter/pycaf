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
    slope_err: float = None
    intercept_err: float = None


class QuadraticFitWithoutSlope(Fit):
    curvature: float
    intercept: float
    curvature_err: float = None
    intercept_err: float = None


class QuadraticFitWithSlope(QuadraticFitWithoutSlope):
    slope: float
    slope_err: float = None


class GaussianFitWithoutOffset(Fit):
    amplitude: float
    centre: float
    width: float
    amplitude_err: float = None
    centre_err: float = None
    width_err: float = None


class GaussianFitWithOffset(GaussianFitWithoutOffset):
    offset: float
    offset_err: float = None


class ExponentialFitWithoutOffset(Fit):
    amplitude: float
    centre: float
    rate: float
    amplitude_err: float = None
    centre_err: float = None
    rate_err: float = None


class ExponentialFitWithOffset(ExponentialFitWithoutOffset):
    offset: float
    offset_err: float = None


class GaussianFitWithoutOffset2D(Fit2D):
    amplitude: float
    centre: Tuple[float, float]
    width: Tuple[float, float]
    amplitude_err: float = None
    centre_err: Tuple[float, float] = None
    width_err: Tuple[float, float] = None


class GaussianFitWithOffset2D(GaussianFitWithoutOffset2D):
    offset: float
    offset_err: float = None


class LorentzianFitWithoutOffset(Fit):
    amplitude: float
    centre: float
    width: float
    amplitude_err: float = None
    centre_err: float = None
    width_err: float = None


class LorentzianFitWithOffset(LorentzianFitWithoutOffset):
    offset: float
    offset_err: float = None


class TrapFrequencyOscillationFit(Fit):
    amplitude: float
    rate: float
    frequency: float
    phase: float
    offset: float
    amplitude_err: float = None
    rate_err: float = None
    frequency_err: float = None
    phase_err: float = None
    offset_err: float = None
