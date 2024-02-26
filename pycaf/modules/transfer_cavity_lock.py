from typing import List, Union, Callable, Dict
import time
import pathlib
from rich.progress import track


def get_laser_set_points_tcl(
    TransferCavityLock,
    lasers: Dict[str, str]
) -> Dict[str, Dict[str, float]]:
    _lasers = {}
    for laser, cavity in lasers.items():
        voltage = TransferCavityLock.GetLaserVoltage(
            cavity, laser
        )
        set_point = TransferCavityLock.GetLaserSetpoint(
            cavity, laser
        )
        _lasers[laser] = {"voltage": voltage, "set_point": set_point}
        print(f"{laser}: voltage = {voltage}, set_point = {set_point}")
    return _lasers


def scan_laser_set_points_tcl(
    Dictionary,
    String,
    Object,
    TransferCavityLock,
    MOTMaster,
    root: pathlib.Path,
    script: str,
    laser: str,
    cavity: str,
    values: List[Union[int, float]],
    pre_callback: Callable = None,
    post_callback: Callable = None,
    motmaster_parameter: str = None,
    motmaster_value: Union[int, float] = None,
    interval: float = 0.05
) -> None:
    _dictionary = Dictionary[String, Object]()
    path = str(root.joinpath(f"{script}.cs"))
    current_set_point = TransferCavityLock.GetLaserSetpoint(
        cavity, laser
    )
    try:
        MOTMaster.SetScriptPath(path)
        if (motmaster_parameter and motmaster_value) is not None:
            _dictionary[motmaster_parameter] = motmaster_value
        while current_set_point > values[0]:
            current_set_point -= 0.001
            TransferCavityLock.SetLaserSetpoint(
                cavity, laser, current_set_point
            )
            time.sleep(interval)
        while current_set_point < values[0]:
            current_set_point += 0.001
            TransferCavityLock.SetLaserSetpoint(
                cavity, laser, current_set_point
            )
            time.sleep(interval)
        for i in track(range(len(values))):
            TransferCavityLock.SetLaserSetpoint(
                cavity, laser, values[i]
            )
            if pre_callback is not None:
                pre_callback()
            MOTMaster.Go(_dictionary)
            if post_callback is not None:
                post_callback()
            time.sleep(interval)
    except Exception as e:
        print(f"Error: {e} encountered")
    return None
