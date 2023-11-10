from typing import Tuple
from scipy.optimize import curve_fit
import numpy as np

from .models import (
    GaussianFitWithOffset,
    GaussianFitWithoutOffset,
    LinearFit,
    ExponentialFitWithOffset,
    ExponentialFitWithoutOffset,
    GaussianFitWithoutOffset2D,
    GaussianFitWithOffset2D
)


def gaussian_with_offset(
    x: float,
    amplitude: float,
    centre: float,
    sigma: float,
    offset: float
) -> float:
    return np.abs(amplitude)*np.exp(-(x-centre)**2/(2*sigma**2))+offset


def fit_gaussian_with_offset(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray = np.array([]),
    n_fine: int = 100
) -> GaussianFitWithOffset:
    loc_trial = np.argmax(y)
    o_trial = np.min(y)
    a_trial = y[loc_trial]
    c_trial = x[loc_trial]
    halfmax_y = np.max(y)/2.0
    s_trial = np.abs(x[0]-x[1])*len(y[y > halfmax_y])/2.0
    p0 = [a_trial, c_trial, s_trial, o_trial]
    popt, _ = curve_fit(gaussian_with_offset, x, y, p0=p0)
    x_fine = np.linspace(np.min(x), np.max(x), n_fine)
    y_fine = gaussian_with_offset(x_fine, *popt)
    fit = GaussianFitWithOffset(
        x=x,
        y=y,
        err=err,
        x_fine=x_fine,
        y_fine=y_fine,
        amplitude=popt[0],
        centre=popt[1],
        width=popt[2],
        offset=popt[3]
    )
    return fit


def gaussian_without_offset(
    x: float,
    amplitude: float,
    centre: float,
    sigma: float
) -> float:
    return np.abs(amplitude)*np.exp(-(x-centre)**2/(2*sigma**2))


def fit_gaussian_without_offset(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray = np.array([]),
    n_fine: int = 100
) -> GaussianFitWithoutOffset:
    loc_trial = np.argmax(y)
    a_trial = y[loc_trial]
    c_trial = x[loc_trial]
    halfmax_y = np.max(y)/2.0
    w_trial = np.abs(x[0]-x[1])*len(y[y > halfmax_y])/2.0
    p0 = [a_trial, c_trial, w_trial]
    popt, _ = curve_fit(gaussian_without_offset, x, y, p0=p0)
    x_fine = np.linspace(np.min(x), np.max(x), n_fine)
    y_fine = gaussian_without_offset(x_fine, *popt)
    fit = GaussianFitWithoutOffset(
        x=x,
        y=y,
        err=err,
        x_fine=x_fine,
        y_fine=y_fine,
        amplitude=popt[0],
        centre=popt[1],
        width=popt[2]
    )
    return fit


def gaussian_without_offset_2D(
    xy: Tuple[np.ndarray, np.ndarray],
    amplitude: float,
    centre: Tuple[float, float],
    sigma: Tuple[float, float],
    theta: float
) -> np.ndarray:
    x, y = xy
    first = (np.cos(theta)**2)/(2*sigma[0]**2) + \
        (np.sin(theta)**2)/(2*sigma[1]**2)*(x-centre[0])**2
    second = -2*(np.sin(2*theta))/(4*sigma[0]**2) + \
        (np.sin(2*theta))/(4*sigma[1]**2)*(x-centre[0])*(y-centre[1])
    third = (np.sin(theta)**2)/(2*sigma[0]**2) + \
        (np.cos(theta)**2)/(2*sigma[1]**2)*(y-centre[1])**2
    total = amplitude*np.exp(-first+second+third)
    return total.ravel()


def fit_gaussian_without_offset_2D(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
) -> GaussianFitWithoutOffset2D:
    amplitude_trial = 0
    centre_trial = (0, 0)
    sigma_trial = (0, 0)
    theta_trial = 0
    mx, my = np.meshgrid(x, y)
    p0 = [amplitude_trial, centre_trial, sigma_trial, theta_trial]
    popt, _ = curve_fit(gaussian_without_offset_2D, (mx, my), data, p0=p0)
    fit = GaussianFitWithoutOffset2D(
        amplitude=popt[0],
        centre=popt[1],
        width=popt[2],
        data=data,
        x=x,
        y=y
    )
    return fit


def gaussian_with_offset_2D(
    x: np.ndarray,
    y: np.ndarray,
    amplitude: float,
    centre: Tuple[float, float],
    sigma: Tuple[float, float],
    offset: float,
    theta: float
) -> None:
    x, y = np.meshgrid(x, y)
    first = (np.cos(theta)**2)/(2*sigma[0]**2) + \
        (np.sin(theta)**2)/(2*sigma[1]**2)*(x-centre[0])**2
    second = -2*(np.sin(2*theta))/(4*sigma[0]**2) + \
        (np.sin(2*theta))/(4*sigma[1]**2)*(x-centre[0])*(y-centre[1])
    third = (np.sin(theta)**2)/(2*sigma[0]**2) + \
        (np.cos(theta)**2)/(2*sigma[1]**2)*(y-centre[1])**2
    total = offset+amplitude*np.exp(-first+second+third)
    return total.ravel()


def fit_gaussian_with_offset_2D(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
) -> GaussianFitWithOffset2D:
    amplitude_trial = 0
    centre_trial = (0, 0)
    sigma_trial = (0, 0)
    offset_trial = 0
    theta_trial = 0
    mx, my = np.meshgrid(x, y)
    p0 = [
        amplitude_trial, centre_trial, sigma_trial,
        offset_trial, theta_trial
    ]
    popt, _ = curve_fit(gaussian_with_offset_2D, (mx, my), data, p0=p0)
    fit = GaussianFitWithOffset2D(
        amplitude=popt[0],
        centre=popt[1],
        width=popt[2],
        offset=popt[3],
        data=data,
        x=x,
        y=y
    )
    return fit


def linear(
    x: float,
    slope: float,
    intercept: float
) -> float:
    return x*slope + intercept


def fit_linear(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray = np.array([]),
    n_fine: int = 100
) -> LinearFit:
    s_trial = (y[-1]-y[0])/(x[-1]-x[0])
    i_trial = np.max(y) if s_trial < 0 else np.min(y)
    p0 = [s_trial, i_trial]
    popt, _ = curve_fit(linear, x, y, p0=p0)
    x_fine = np.linspace(np.min(x), np.max(x), n_fine)
    y_fine = linear(x_fine, *popt)
    fit = LinearFit(
        x=x,
        y=y,
        err=err,
        x_fine=x_fine,
        y_fine=y_fine,
        slope=popt[0],
        intercept=popt[1]
    )
    return fit


def exponential_without_offset(
    x: float,
    amplitude: float,
    centre: float,
    rate: float
) -> float:
    return amplitude*np.exp(-(x-centre)/rate)


def fit_exponential_without_offset(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray = np.array([]),
    n_fine: int = 100
) -> ExponentialFitWithoutOffset:
    a_trial = np.max(y)
    c_trial = x[np.argmax(y)]
    r_trial = np.nanmean((c_trial-x)/np.log(y/a_trial))
    p0 = [a_trial, c_trial, r_trial]
    popt, _ = curve_fit(exponential_without_offset, x, y, p0=p0)
    x_fine = np.linspace(np.min(x), np.max(x), n_fine)
    y_fine = exponential_without_offset(x_fine, *popt)
    fit = ExponentialFitWithoutOffset(
        x=x,
        y=y,
        err=err,
        x_fine=x_fine,
        y_fine=y_fine,
        amplitude=popt[0],
        centre=popt[1],
        rate=popt[2]
    )
    return fit


def exponential_with_offset(
    x: float,
    amplitude: float,
    centre: float,
    rate: float,
    offset: float
) -> float:
    return amplitude*np.exp(-(x-centre)/rate)+offset


def fit_exponential_with_offset(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray = np.array([]),
    n_fine: int = 100
) -> ExponentialFitWithOffset:
    a_trial = np.max(y)
    c_trial = x[np.argmax(y)]
    r_trial = np.nanmean((c_trial-x)/np.log(y/a_trial))
    o_trial = np.min(y)
    p0 = [a_trial, c_trial, r_trial, o_trial]
    popt, _ = curve_fit(exponential_with_offset, x, y, p0=p0)
    x_fine = np.linspace(np.min(x), np.max(x), n_fine)
    y_fine = exponential_with_offset(x_fine, *popt)
    fit = ExponentialFitWithOffset(
        x=x,
        y=y,
        err=err,
        x_fine=x_fine,
        y_fine=y_fine,
        amplitude=popt[0],
        centre=popt[1],
        rate=popt[2],
        offset=popt[3]
    )
    return fit
