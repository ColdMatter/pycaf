from typing import Any, Callable, Dict, List, Union, Tuple
from itertools import product
from rich.progress import track
import datetime
import numpy as np
import pathlib
import json
import sys
import time

# NOTE: these imports will only work with the pythonnet package
try:
    import clr
    from System.Collections.Generic import Dictionary
    from System import String, Object, Int32
    from System import Activator
except Exception as e:
    print(f"Error: {e} encountered, probably no pythonnet")

from pycaf.modules import (
    PicoMotor8742
)


def get_laser_frequencies(
    WavemeterLock,
    lasers: List[str]
) -> List[str]:
    _lasers = {}
    for laser in lasers:
        set_point = WavemeterLock.getSlaveFrequency(laser)
        _lasers[laser] = set_point
        print(f"Frequency of {laser}: {set_point:.6f} THz")
    return _lasers


def save_laser_frequencies(
    WavemeterLock,
    lasers: List[str],
    dirpath: str
) -> None:
    filename = "wavemeterlock.txt"
    filepath = pathlib.Path(dirpath).joinpath(filename)
    freq = {}
    for laser in lasers:
        freq[f"{laser}_set"] = \
            int(np.ceil(float(WavemeterLock.getSetFrequency(laser))*1e6))/1e6
        freq[f"{laser}_act"] = \
            int(np.ceil(float(WavemeterLock.getSlaveFrequency(laser))*1e6))/1e6
    with open(filepath, "w") as f:
        f.write(json.dumps(freq))
    return None


def move_laser_frequency(
    WavemeterLock,
    laser: str,
    final_set_point: float
) -> None:
    WavemeterLock.setSlaveFrequency(
        laser,
        int(np.ceil(float(final_set_point)*1e6))/1e6
    )
    return None


def run(
    Dictionary,
    String,
    Object,
    WavemeterLock,
    MOTMaster,
    lasers: List[str],
    root: pathlib.Path,
    wavemeter_info_dirpath: pathlib.Path,
    script: str,
    state_dims: List[str],
    state_value: List[Union[int, float]],
    pre_callback: Callable = None,
    post_callback: Callable = None,
    interval: float = 0.01,
    **kwargs
) -> None:
    _dictionary = Dictionary[String, Object]()
    path = str(root.joinpath(f"{script}.cs"))
    try:
        MOTMaster.SetScriptPath(path)
    except Exception as e:
        print(f"Error: {e} encountered")
    for i, state in enumerate(state_dims):
        if state[0:9] == "wavemeter":
            move_laser_frequency(
                WavemeterLock,
                state[10:],
                state_value[i]
            )
        if state[0:9] == "motmaster":
            _value = state_value[i]
            _value_type = type(_value)
            if _value_type == int:
                _dictionary[state[10:]] = Int32(state_value[i])
            else:
                _dictionary[state[10:]] = state_value[i]
    save_laser_frequencies(
        WavemeterLock,
        lasers,
        wavemeter_info_dirpath
    )
    if pre_callback is not None:
        pre_callback(state_dims, state_value, **kwargs)
    try:
        MOTMaster.Go(_dictionary)
    except Exception as e:
        print(f"Error: {e} encountered")
    if post_callback is not None:
        post_callback(state_dims, state_value, **kwargs)
    time.sleep(interval)
    return None


def scan(
    Dictionary,
    String,
    Object,
    WavemeterLock,
    MOTMaster,
    lasers: List[str],
    root: pathlib.Path,
    wavemeter_info_dirpath: pathlib.Path,
    script: str,
    motmaster_parameters_with_values: Dict[str, List[Union[int, float]]],
    lasers_with_frequencies: Dict[str, List[float]] = None,
    n_iter: int = 1,
    pre_callback: Callable = None,
    post_callback: Callable = None,
    interval: float = 0.01,
    **kwargs
) -> None:
    state_dict = {}
    if lasers_with_frequencies is not None:
        for key, value in lasers_with_frequencies.items():
            state_dict[f"wavemeter_{key}"] = value
    for key, value in motmaster_parameters_with_values.items():
        state_dict[f"motmaster_{key}"] = value
    state_dims = list(state_dict.keys())
    _state_space = []
    for item in product(*list(state_dict.values())):
        _state_space.append(item)
    state_space = []
    for _ in range(n_iter):
        state_space.extend(_state_space)
    for state_value in track(state_space):
        run(
            Dictionary, String, Object,
            WavemeterLock, MOTMaster,
            lasers, root, wavemeter_info_dirpath, script,
            state_dims, state_value,
            pre_callback, post_callback, interval,
            **kwargs
        )
    return None


class Experiment():
    def __init__(
        self,
        config_path: str,
        interval: Union[int, float]
    ) -> None:
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.root = pathlib.Path(self.config["script_root_path"])
        self.interval = interval
        self.edmsuite_modules = self.config["edmsuite_modules"]
        self.edmsuite_dlls = self.config["edmsuite_dlls"]
        self.default_remote_path_id = self.config["default_remote_path_id"]
        self.wavemeter_info_path = self.config["temp_wavemeter_info_path"]
        self.today = datetime.date.today()
        self.data_prefix = self.config["data_prefix"]
        for path in self.edmsuite_dlls.values():
            self._add_ref(path)
        for key, info in self.edmsuite_modules.items():
            self._add_ref(info["exe_path"])
            try:
                _module = __import__(key)
                for id, remote_path in info["remote_paths"].items():
                    self.__dict__[f"{key}_{id}"] = Activator.GetObject(
                        _module.Controller,
                        remote_path
                    )
            except Exception as e:
                print(f"Error: {e} encountered")

    def _add_ref(
        self,
        path: str
    ) -> None:
        _path = pathlib.Path(path)
        sys.path.append(_path.parent)
        clr.AddReference(path)
        return None

    def __setattr__(
        self,
        __name: str,
        __value: Any
    ) -> None:
        self.__dict__[__name] = __value

    def connect_picomotor_plugin(
        self
    ) -> None:
        try:
            self.picomotor = PicoMotor8742(
                self.config, self.interval,
                Dictionary, String, Object
            )
        except Exception as e:
            print(f"Error: {e} encountered")
        return None

    def connect_picoscope_plugin(
        self
    ) -> None:
        return None

    def connect_evalboard_ad9959_plugin(
        self
    ) -> None:
        return None

    def get_remote_path_ids(
        self,
        **kwargs
    ) -> Tuple[str, str]:
        if "motmaster_remote_path_id" in kwargs:
            motmaster_remote_path_id = kwargs["motmaster_remote_path_id"]
        else:
            motmaster_remote_path_id = self.default_remote_path_id
        if "wavemeter_remote_path_id" in kwargs:
            wavemeter_remote_path_id = kwargs["wavemeter_remote_path_id"]
        else:
            wavemeter_remote_path_id = self.default_remote_path_id
        return motmaster_remote_path_id, wavemeter_remote_path_id

    def scan(
        self,
        script: str,
        motmaster_parameters_with_values: Dict[str, List[Union[int, float]]],
        lasers_with_frequencies: Dict[str, List[float]] = None,
        n_iter: int = 1,
        pre_callback: Callable = None,
        post_callback: Callable = None,
        **kwargs
    ) -> None:
        mmrp_id, wmrp_id = self.get_remote_path_ids(**kwargs)
        _motmaster = self.__dict__[f"MOTMaster_{mmrp_id}"]
        _wavemeterlock = self.__dict__[f"WavemeterLock_{wmrp_id}"]
        _lasers = self.config["edmsuite_modules"]["WavemeterLock"]["lasers"]
        scan(
            Dictionary, String, Object,
            _wavemeterlock, _motmaster, _lasers[wmrp_id],
            self.root, self.wavemeter_info_path, script,
            motmaster_parameters_with_values, lasers_with_frequencies,
            n_iter, pre_callback, post_callback, self.interval,
            **kwargs
        )
        return None
