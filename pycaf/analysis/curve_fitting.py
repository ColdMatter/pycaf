from typing import Tuple
from scipy.optimize import curve_fit
import numpy as np

from .models import (
    LinearFit,
    QuadraticFitWithoutSlope,
    GaussianFitWithOffset,
    GaussianFitWithoutOffset,
    ExponentialFitWithOffset,
    ExponentialFitWithoutOffset,
    GaussianFitWithoutOffset2D,
    GaussianFitWithOffset2D,
    LorentzianFitWithOffset,
    LorentzianFitWithoutOffset,
    TrapFrequencyOscillationFit
)


def linear(
    x: float,
    slope: float,
    intercept: float
) -> float:
    return x*slope + intercept


def fit_linear(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray = None,
    n_fine: int = 100
) -> LinearFit:
    s_trial = (y[-1]-y[0])/(x[-1]-x[0])
    i_trial = np.max(y) if s_trial < 0 else np.min(y)
    p0 = [s_trial, i_trial]
    fit, popt = None, None
    try:
        popt, _ = curve_fit(linear, x, y, p0=p0, sigma=err)
    except Exception as e:
        print(f"Error {e} occured during fitting")
    if popt is not None:
        x_fine = np.linspace(np.min(x), np.max(x), n_fine)
        y_fine = linear(x_fine, *popt)
        func_str = "\n y = m*x+c"
        args_str = f"\n m: {popt[0]:e}\n c: {popt[1]:e}"
        fit = LinearFit(
            func_str=func_str,
            args_str=args_str,
            x=x,
            y=y,
            err=err,
            x_fine=x_fine,
            y_fine=y_fine,
            slope=popt[0],
            intercept=popt[1]
        )
    return fit


def quadratic_without_slope(
    x: float,
    curvature: float,
    intercept: float
) -> float:
    return x**2*curvature + intercept


def fit_quadratic_without_slope(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray = None,
    n_fine: int = 100
) -> QuadraticFitWithoutSlope:
    c_trial = 0.0
    s_trial = (y[-1]-y[0])/(x[-1]-x[0])
    i_trial = np.max(y) if s_trial < 0 else np.min(y)
    p0 = [c_trial, i_trial]
    fit, popt = None, None
    try:
        popt, _ = curve_fit(quadratic_without_slope, x, y, p0=p0, sigma=err)
    except Exception as e:
        print(f"Error {e} occured during fitting")
    if popt is not None:
        x_fine = np.linspace(np.min(x), np.max(x), n_fine)
        y_fine = quadratic_without_slope(x_fine, *popt)
        func_str = "\n y = a*x^2+c"
        args_str = f"\n a: {popt[0]:e}\n c: {popt[1]:e}"
        fit = QuadraticFitWithoutSlope(
            func_str=func_str,
            args_str=args_str,
            x=x,
            y=y,
            err=err,
            x_fine=x_fine,
            y_fine=y_fine,
            curvature=popt[0],
            intercept=popt[1]
        )
    return fit


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
    err: np.ndarray = None,
    n_fine: int = 100
) -> GaussianFitWithOffset:
    loc_trial = np.argmax(y)
    o_trial = np.min(y)
    a_trial = y[loc_trial]
    c_trial = x[loc_trial]
    halfmax_y = np.max(y)/2.0
    s_trial = np.abs(x[0]-x[1])*len(y[y > halfmax_y])/2.0
    p0 = [a_trial, c_trial, s_trial, o_trial]
    fit, popt = None, None
    try:
        popt, _ = curve_fit(gaussian_with_offset, x, y, p0=p0, sigma=err)
    except Exception as e:
        print(f"Error {e} occured during fitting")
    if popt is not None:
        x_fine = np.linspace(np.min(x), np.max(x), n_fine)
        y_fine = gaussian_with_offset(x_fine, *popt)
        func_str = "\n y = a*exp(-(x-xc)**2/(2*s**2))+o"
        args_str = f"\n a: {popt[0]:e}\n xc: {popt[1]:e}" + \
            f"\n s: {np.abs(popt[2]):e}\n o: {popt[3]:e}"
        fit = GaussianFitWithOffset(
            func_str=func_str,
            args_str=args_str,
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
    err: np.ndarray = None,
    n_fine: int = 100
) -> GaussianFitWithoutOffset:
    loc_trial = np.argmax(y)
    a_trial = y[loc_trial]
    c_trial = x[loc_trial]
    halfmax_y = np.max(y)/2.0
    w_trial = np.abs(x[0]-x[1])*len(y[y > halfmax_y])/2.0
    p0 = [a_trial, c_trial, w_trial]
    fit, popt = None, None
    try:
        popt, _ = curve_fit(gaussian_without_offset, x, y, p0=p0, sigma=err)
    except Exception as e:
        print(f"Error {e} occured during fitting")
    if popt is not None:
        x_fine = np.linspace(np.min(x), np.max(x), n_fine)
        y_fine = gaussian_without_offset(x_fine, *popt)
        func_str = "\n y = a*exp(-(x-xc)**2/(2*s**2))"
        args_str = \
            f"\n a: {popt[0]:e}\n xc: {popt[1]:e}\n s: {np.abs(popt[2]):e}"
        fit = GaussianFitWithoutOffset(
            func_str=func_str,
            args_str=args_str,
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
    centre_trial = []
    sigma_trial = []
    theta_trial = 0
    for i, axis in zip(range(2), [x, y]):
        _sum = np.sum(data, axis=i)
        loc_trial = np.argmax(_sum)
        a_trial = _sum[loc_trial]
        c_trial = axis[loc_trial]
        halfmax_y = np.max(_sum)/2.0
        s_trial = np.abs(axis[0]-axis[1])*len(_sum[_sum > halfmax_y])/2.0
        if amplitude_trial < a_trial:
            amplitude_trial = a_trial
        centre_trial.append(c_trial)
        sigma_trial.append(s_trial)
    mx, my = np.meshgrid(x, y)
    p0 = [
        amplitude_trial, tuple(centre_trial),
        tuple(sigma_trial), theta_trial
    ]
    popt, _ = curve_fit(gaussian_without_offset_2D, (mx, my), data, p0=p0)
    func_str = "\n y = a*exp(-((x-xc)**2/(2*xs**2)+(y-yc)**2/(2*ys**2)))"
    args_str = f"\n a: {popt[0]:e}\n (xc, yc): {popt[1]:e}" + \
        f"\n (xs, ys): {popt[2]:e}\n"
    fit = GaussianFitWithoutOffset2D(
        func_str=func_str,
        args_str=args_str,
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
    centre_trial = []
    sigma_trial = []
    theta_trial = 0
    offset_trial = np.min(data)
    for i, axis in zip(range(2), [x, y]):
        _sum = np.sum(data, axis=i)
        loc_trial = np.argmax(_sum)
        a_trial = _sum[loc_trial]
        c_trial = axis[loc_trial]
        halfmax_y = np.max(_sum)/2.0
        s_trial = np.abs(axis[0]-axis[1])*len(_sum[_sum > halfmax_y])/2.0
        if amplitude_trial < a_trial:
            amplitude_trial = a_trial
        centre_trial.append(c_trial)
        sigma_trial.append(s_trial)
    mx, my = np.meshgrid(x, y)
    p0 = [
        amplitude_trial, tuple(centre_trial),
        tuple(sigma_trial), offset_trial, theta_trial
    ]
    popt, _ = curve_fit(gaussian_with_offset_2D, (mx, my), data, p0=p0)
    func_str = "\n y = a*exp(-((x-xc)**2/(2*xs**2)+(y-yc)**2/(2*ys**2)))+o"
    args_str = f"\n a: {popt[0]:e}\n (xc, yc): {popt[1]:e}" + \
        f"\n (xs, ys): {popt[2]:e}\n o: {popt[3]:e}"
    fit = GaussianFitWithOffset2D(
        func_str=func_str,
        args_str=args_str,
        amplitude=popt[0],
        centre=popt[1],
        width=popt[2],
        offset=popt[3],
        data=data,
        x=x,
        y=y
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
    err: np.ndarray = None,
    n_fine: int = 100
) -> ExponentialFitWithoutOffset:
    a_trial = np.max(y)
    c_trial = x[np.argmax(y)]
    r_trial = np.nanmean((c_trial-x)/np.log(y/a_trial))
    p0 = [a_trial, c_trial, r_trial]
    fit, popt = None, None
    try:
        popt, _ = curve_fit(exponential_without_offset, x, y, p0=p0, sigma=err)
    except Exception as e:
        print(f"Error {e} occured during fitting")
    if popt is not None:
        x_fine = np.linspace(np.min(x), np.max(x), n_fine)
        y_fine = exponential_without_offset(x_fine, *popt)
        func_str = "\n y = a*exp(-(x-xc)/r)"
        args_str = f"\n a: {popt[0]:e}\n xc: {popt[1]:e}" + \
            f"\n r: {popt[2]:e}"
        fit = ExponentialFitWithoutOffset(
            func_str=func_str,
            args_str=args_str,
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
    err: np.ndarray = None,
    n_fine: int = 100
) -> ExponentialFitWithOffset:
    a_trial = np.max(y)
    c_trial = x[np.argmax(y)]
    r_trial = np.nanmean((c_trial-x)/np.log(y/a_trial))
    o_trial = np.min(y)
    p0 = [a_trial, c_trial, r_trial, o_trial]
    fit, popt = None, None
    try:
        popt, _ = curve_fit(exponential_with_offset, x, y, p0=p0, sigma=err)
    except Exception as e:
        print(f"Error {e} occured during fitting")
    if popt is not None:
        x_fine = np.linspace(np.min(x), np.max(x), n_fine)
        y_fine = exponential_with_offset(x_fine, *popt)
        func_str = "\n y = a*exp(-(x-xc)/r)+o"
        args_str = f"\n a: {popt[0]:e}\n xc: {popt[1]:e}" + \
            f"\n r: {popt[2]:e}\n o: {popt[3]:e}"
        fit = ExponentialFitWithOffset(
            func_str=func_str,
            args_str=args_str,
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


def lorentzian_without_offset(
    x: float,
    amplitude: float,
    centre: float,
    width: float
) -> float:
    return amplitude*(0.5*width**2)/((x-centre)**2+(0.5*width**2))


def fit_lorentzian_without_offset(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray = None,
    n_fine: int = 100
) -> LorentzianFitWithoutOffset:
    a_trial = np.max(y)
    c_trial = x[np.argmax(y)]
    r_trial = np.nanmean(x/(np.sqrt(a_trial/y-1)))
    p0 = [a_trial, c_trial, r_trial]
    fit, popt = None, None
    try:
        popt, _ = curve_fit(lorentzian_without_offset, x, y, p0=p0, sigma=err)
    except Exception as e:
        print(f"Error {e} occured during fitting")
    if popt is not None:
        x_fine = np.linspace(np.min(x), np.max(x), n_fine)
        y_fine = lorentzian_without_offset(x_fine, *popt)
        func_str = "\n y = a*(0.5*w**2/((x-c)**2+(0.5*w**2)))"
        args_str = f"\n a: {popt[0]:e}\n xc: {popt[1]:e}" + \
            f"\n w: {np.abs(popt[2]):e}"
        fit = LorentzianFitWithoutOffset(
            func_str=func_str,
            args_str=args_str,
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


def lorentzian_with_offset(
    x: float,
    amplitude: float,
    centre: float,
    width: float,
    offset: float
) -> float:
    return amplitude*(0.5*width**2)/((x-centre)**2+(0.5*width**2))+offset


def fit_lorentzian_with_offset(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray = None,
    n_fine: int = 100
) -> LorentzianFitWithOffset:
    a_trial = np.max(y)
    c_trial = x[np.argmax(y)]
    o_trial = np.min(y)
    r_trial = np.nanmean(x/(np.sqrt(a_trial/(y-o_trial)-1)))
    p0 = [a_trial, c_trial, r_trial, o_trial]
    fit, popt = None, None
    try:
        popt, _ = curve_fit(lorentzian_with_offset, x, y, p0=p0, sigma=err)
    except Exception as e:
        print(f"Error {e} occured during fitting")
    if popt is not None:
        x_fine = np.linspace(np.min(x), np.max(x), n_fine)
        y_fine = lorentzian_with_offset(x_fine, *popt)
        func_str = "\n y = a*(0.5*w**2/((x-xc)**2+(0.5*w**2)))+o"
        args_str = f"\n a: {popt[0]:e}\n xc: {popt[1]:e}" + \
            f"\n w: {np.abs(popt[2]):e}\n o: {popt[3]:e}"
        fit = LorentzianFitWithOffset(
            func_str=func_str,
            args_str=args_str,
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


def trap_frequency_oscillation(
    x: float,
    amplitude: float,
    rate: float,
    phase: float,
    frequency: float,
    offset: float
) -> float:
    cosine_term = np.cos(np.sqrt((2*np.pi*frequency)**2-0.25*rate**2)*x-phase)
    return amplitude*np.exp(-0.5*rate*x)*cosine_term+offset


def fit_trap_frequency_oscillation(
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray = None,
    n_fine: int = 100
) -> TrapFrequencyOscillationFit:
    a_trial = np.max(y)
    r_trial = 0.0
    p_trial = x[np.argmax(y)]
    f_trial = np.min(y)
    o_trial = np.mean(y)
    p0 = [a_trial, r_trial, p_trial, f_trial, o_trial]
    fit, popt = None, None
    try:
        popt, _ = curve_fit(trap_frequency_oscillation, x, y)
    except Exception as e:
        print(f"Error {e} occured during fitting")
    if popt is not None:
        x_fine = np.linspace(np.min(x), np.max(x), n_fine)
        y_fine = trap_frequency_oscillation(x_fine, *popt)
        func_str = "\n y = a*exp(-r*x/2)*cos(.....)"
        args_str = f"\n a: {popt[0]:e}\n r: {popt[1]:e}" + \
            f"\n p: {np.abs(popt[2]):e}\n f: {popt[3]:e}" + \
            f"\n o: {popt[4]:e}\n"
        fit = TrapFrequencyOscillationFit(
            func_str=func_str,
            args_str=args_str,
            x=x,
            y=y,
            err=err,
            x_fine=x_fine,
            y_fine=y_fine,
            amplitude=popt[0],
            rate=popt[1],
            phase=popt[2],
            frequency=popt[3],
            offset=popt[4]
        )
    return fit
