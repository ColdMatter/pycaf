from typing import List, Union, Callable, Tuple
import time
import pathlib
from rich.progress import track


def get_laser_set_points(
    WavemeterLock,
    lasers: List[str]
) -> List[str]:
    _lasers = {}
    for laser in lasers:
        set_point = WavemeterLock.getSlaveFrequency(laser)
        _lasers[laser] = set_point
        print(f"Frequency of {laser}: {set_point:.6f} THz")
    return _lasers


def scan_laser_set_points(
    Dictionary,
    String,
    Object,
    WavemeterLock,
    MOTMaster,
    root: pathlib.Path,
    script: str,
    laser: str,
    values: List[Union[int, float]],
    pre_callback: Callable = None,
    post_callback: Callable = None,
    motmaster_parameter: str = None,
    motmaster_value: Union[int, float] = None,
    interval: float = 0.05
) -> None:
    _dictionary = Dictionary[String, Object]()
    path = str(root.joinpath(f"{script}.cs"))
    current_set_point = float(WavemeterLock.getSlaveFrequency(
        laser
    ))
    try:
        MOTMaster.SetScriptPath(path)
        if (motmaster_parameter and motmaster_value) is not None:
            _dictionary[motmaster_parameter] = motmaster_value
        while current_set_point > values[0]:
            current_set_point -= 0.00001
            WavemeterLock.setSlaveFrequency(
                laser, current_set_point
            )
            time.sleep(interval)
        while current_set_point < values[0]:
            current_set_point += 0.00001
            WavemeterLock.setSlaveFrequency(
                laser, current_set_point
            )
            time.sleep(interval)
        for i in track(range(len(values))):
            if pre_callback is not None:
                pre_callback()
            WavemeterLock.setSlaveFrequency(
                laser, values[i]
            )
            MOTMaster.Go(_dictionary)
            if post_callback is not None:
                post_callback()
            time.sleep(interval)
    except Exception as e:
        print(f"Error: {e} encountered")
    return None


def scan_laser_set_points_with_motmaster_values(
    Dictionary,
    String,
    Object,
    WavemeterLock,
    MOTMaster,
    root: pathlib.Path,
    script: str,
    laser: str,
    values: List[Union[int, float]],
    pre_callback: Callable = None,
    post_callback: Callable = None,
    motmaster_parameter: str = None,
    motmaster_values: List[Union[int, float]] = None,
    interval: float = 0.05
) -> None:
    _dictionary = Dictionary[String, Object]()
    path = str(root.joinpath(f"{script}.cs"))
    current_set_point = float(WavemeterLock.getSlaveFrequency(
        laser
    ))
    try:
        MOTMaster.SetScriptPath(path)
        while current_set_point > values[0]:
            current_set_point -= 0.00001
            WavemeterLock.setSlaveFrequency(
                laser, current_set_point
            )
            time.sleep(interval)
        while current_set_point < values[0]:
            current_set_point += 0.00001
            WavemeterLock.setSlaveFrequency(
                laser, current_set_point
            )
            time.sleep(interval)
        for i in track(range(len(values))):
            WavemeterLock.setSlaveFrequency(
                laser, values[i]
            )
            if (motmaster_parameter and motmaster_values) is not None:
                for k in range(len(motmaster_values)):
                    _dictionary[motmaster_parameter] = motmaster_values[k]
                    if pre_callback is not None:
                        pre_callback()
                    MOTMaster.Go(_dictionary)
                    time.sleep(interval)
                    if post_callback is not None:
                        post_callback()
            else:
                if pre_callback is not None:
                    pre_callback()
                MOTMaster.Go(_dictionary)
                time.sleep(interval)
                if post_callback is not None:
                    post_callback()
    except Exception as e:
        print(f"Error: {e} encountered")
    return None


def scan_laser_set_points_with_motmaster_multiple_parameters(
    Dictionary,
    String,
    Object,
    WavemeterLock,
    MOTMaster,
    root: str,
    script: str,
    laser: str,
    values: List[Union[int, float]],
    pre_callback: Callable = None,
    post_callback: Callable = None,
    motmaster_parameters: List[str] = None,
    motmaster_values: List[Tuple[Union[int, float]]] = None,
    interval: float = 0.5
) -> None:
    _dictionary = Dictionary[String, Object]()
    path = str(root.joinpath(f"{script}.cs"))
    current_set_point = float(WavemeterLock.getSlaveFrequency(
        laser
    ))
    try:
        MOTMaster.SetScriptPath(path)
        while current_set_point > values[0]:
            current_set_point -= 0.00001
            WavemeterLock.setSlaveFrequency(
                laser, current_set_point
            )
            time.sleep(0.05)
        while current_set_point < values[0]:
            current_set_point += 0.00001
            WavemeterLock.setSlaveFrequency(
                laser, current_set_point
            )
            time.sleep(0.05)
        for i in track(range(len(values))):
            WavemeterLock.setSlaveFrequency(
                laser, values[i]
            )
            if (motmaster_parameters and motmaster_values) is not None:
                for k in range(len(motmaster_values)):
                    motmaster_value: Tuple = motmaster_values[k]
                    for t, parameter in enumerate(motmaster_parameters):
                        _dictionary[parameter] = motmaster_value[t]
                    if pre_callback is not None:
                        pre_callback()
                    MOTMaster.Go(_dictionary)
                    time.sleep(interval)
                    if post_callback is not None:
                        post_callback()
            else:
                if pre_callback is not None:
                    pre_callback()
                MOTMaster.Go(_dictionary)
                time.sleep(interval)
                if post_callback is not None:
                    post_callback()
    except Exception as e:
        print(f"Error: {e} encountered")
    return None
