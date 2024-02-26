from typing import List, Union, Callable, Any, Tuple
import time
import pathlib
from rich.progress import track


def single_run(
    Dictionary,
    String,
    Object,
    MOTMaster,
    root: pathlib.Path,
    script: str,
    parameter: str = "",
    value: Union[int, float] = None,
    pre_callback: Callable = None,
    post_callback: Callable = None,
    interval: float = 0.05
) -> None:
    _dictionary = Dictionary[String, Object]()
    if len(parameter):
        _dictionary[parameter] = value
    path = str(root.joinpath(f"{script}.cs"))
    try:
        if pre_callback is not None:
            pre_callback()
        MOTMaster.SetScriptPath(path)
        MOTMaster.Go(_dictionary)
        if post_callback is not None:
            post_callback()
        time.sleep(interval)
    except Exception as e:
        print(f"Error: {e} encountered")
    return None


def scan_parameter(
    Dictionary,
    String,
    Object,
    MOTMaster,
    root: pathlib.Path,
    script: str,
    parameter: str,
    values: List[Union[int, float]],
    pre_callback: Callable = None,
    post_callback: Callable = None,
    interval: float = 0.05
) -> None:
    _dictionary = Dictionary[String, Object]()
    path = str(root.joinpath(f"{script}.cs"))
    try:
        MOTMaster.SetScriptPath(path)
        for i in track(range(len(values))):
            _dictionary[parameter] = values[i]
            if pre_callback is not None:
                pre_callback()
            MOTMaster.Go(_dictionary)
            if post_callback is not None:
                post_callback()
            time.sleep(interval)
    except Exception as e:
        print(f"Error: {e} encountered")
    return None


def scan_parameters(
    Dictionary,
    String,
    Object,
    MOTMaster,
    root: pathlib.Path,
    script: str,
    parameters: List[str],
    values: List[Tuple[Any]],
    pre_callback: Callable = None,
    post_callback: Callable = None,
    interval: float = 0.05
) -> None:
    _dictionary = Dictionary[String, Object]()
    path = str(root.joinpath(f"{script}.cs"))
    try:
        MOTMaster.SetScriptPath(path)
        for i in track(range(len(values))):
            if pre_callback is not None:
                pre_callback()
            for k, parameter in enumerate(parameters):
                _dictionary[parameter] = values[i][k]
            MOTMaster.Go(_dictionary)
            if post_callback is not None:
                post_callback()
            time.sleep(interval)
    except Exception as e:
        print(f"Error: {e} encountered")
    return None
