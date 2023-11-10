from typing import Dict, Union, Tuple
from pydantic import BaseModel
import numpy as np


class Pattern(BaseModel):
    name: str
    time: np.ndarray
    event: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class Fit(BaseModel):
    x: np.ndarray = np.array([])
    y: np.ndarray = np.array([])
    err: np.ndarray = np.array([])
    x_fine: np.ndarray = np.array([])
    y_fine: np.ndarray = np.array([])

    class Config:
        arbitrary_types_allowed = True


class Fit2D(BaseModel):
    data: np.ndarray = np.array([[]])
    err: np.ndarray = np.array([[]])

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


class Study(BaseModel):
    cloud: str = None
    n_trigger: int = None
    repetition_rate: int = None
    repetition_skip: int = None
    detection_type: str = None
    info: Dict = dict()
    images: np.ndarray = None
    files: np.ndarray = None
    background_files: np.ndarray = None
    parameters: np.ndarray = None
    numbers: np.ndarray = None
    horizontal_fits: np.ndarray = None
    horizontal_fits_std_error: np.ndarray = None
    vertical_fits: np.ndarray = None
    vertical_fits_std_error: np.ndarray = None
    digital_patterns: Dict[str, Dict[str, Pattern]] = dict()
    analog_patterns: Dict[str, Dict[str, Pattern]] = dict()
    fit: Union[
        LinearFit,
        GaussianFitWithOffset,
        GaussianFitWithoutOffset,
        ExponentialFitWithOffset,
        ExponentialFitWithoutOffset
    ] = None

    class Config:
        arbitrary_types_allowed = True
