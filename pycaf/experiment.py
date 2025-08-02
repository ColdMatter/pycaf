from typing import Any, Callable, Dict, List, Union, Tuple
from itertools import product
from rich.progress import track
import datetime
import numpy as np
import pathlib
import json
import sys
import time
import ast

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

from pycaf.modules import set_evalboard


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


def set_gigatrons_frequency(
    CaFBECHardwareController,
    freq: float
) -> float:
    CaFBECHardwareController.SetGigatronicsFrequency(freq)
    return None


def set_gigatrons_amplitude(
    CaFBECHardwareController,
    amp: float
) -> float:
    CaFBECHardwareController.SetGigatronicsAmplitude(amp)
    return None


def save_gigatrons_parameters(
    CaFBECHardwareController,
    dirpath: str
) -> None:
    filename = "gigatrons_params.txt"
    filepath = pathlib.Path(dirpath).joinpath(filename)
    freq = CaFBECHardwareController.GetGigatronicsFrequency()
    amp = CaFBECHardwareController.GetGigatronicsAmplitude()
    write = {
        "frequency": freq,
        "amplitude": amp
    }
    with open(filepath, "w") as f:
        f.write(json.dumps(write))
    return None


def set_evalboard_settings(
    config_path: str,
    settings: str,
    dirpath: str
) -> None:
    filename = "evalboard.txt"
    filepath = pathlib.Path(dirpath).joinpath(filename)
    parsed_settings = ast.literal_eval(str(settings))
    set_evalboard(config_path, **parsed_settings)
    with open(filepath, "w") as f:
        f.write(json.dumps(settings))
    return None

def run(
    Dictionary,
    String,
    Object,
    CaFBECHardwareController,
    WavemeterLock,
    MOTMaster,
    lasers: List[str],
    root: pathlib.Path,
    wavemeter_info_dirpath: pathlib.Path,
    script: str,
    state_dims: List[str],
    state_value: List[Union[int, float]],
    config_path: str,
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
        if state[0:9] == "gigatrons":
            if state[10:] == "frequency":
                set_gigatrons_frequency(
                    CaFBECHardwareController,
                    state_value[i]
                )
            elif state[10:] == "amplitude":
                set_gigatrons_amplitude(
                    CaFBECHardwareController,
                    state_value[i]
                )
            else:
                print("Error: gigatrons parameter not recognised.")
        if state[0:9] == "evalboard":
            set_evalboard_settings(
                config_path,
                state_value[i],
                wavemeter_info_dirpath
            )
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
    save_gigatrons_parameters(
        CaFBECHardwareController,
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
    CaFBECHardwareController,
    WavemeterLock,
    MOTMaster,
    lasers: List[str],
    root: pathlib.Path,
    wavemeter_info_dirpath: pathlib.Path,
    script: str,
    motmaster_parameters_with_values: Dict[str, List[Union[int, float]]],
    config_path: str,
    lasers_with_frequencies: Dict[str, List[float]] = None,
    n_iter: int = 1,
    pre_callback: Callable = None,
    post_callback: Callable = None,
    interval: float = 0.01,
    motmaster_parameter_doubles_with_values: Dict[str, List[Union[int, float]]] = None,
    gigatrons: Dict[str, List[float]] = None,
    evalboard: List[Dict[str, Any]] = None,
    **kwargs
) -> None:
    state_dict = {}
    if gigatrons is not None:
        for key, value in gigatrons.items():
            state_dict[f"gigatrons_{key}"] = value
    if lasers_with_frequencies is not None:
        for key, value in lasers_with_frequencies.items():
            state_dict[f"wavemeter_{key}"] = value

    if evalboard is not None:
        _evalboard = []
        for item in evalboard:
            _evalboard.append(str(item))
        state_dict["evalboard"] = _evalboard

    # for key, value in motmaster_parameters_with_values.items():
    #     state_dict[f"motmaster_{key}"] = value
    # state_dims = list(state_dict.keys())
    # _state_space = []
    # for item in product(*list(state_dict.values())):
    #     _state_space.append(item)

    state_dict_zip = {}  
    if motmaster_parameter_doubles_with_values is not None:
        for key, value in motmaster_parameter_doubles_with_values.items():
            state_dict_zip[f"motmaster_{key}"] = value
    for key, value in motmaster_parameters_with_values.items():
        state_dict[f"motmaster_{key}"] = value
    state_dims = list(state_dict_zip.keys()) + list(state_dict.keys())
    _state_space = []
    if motmaster_parameter_doubles_with_values is not None:
        for item in product(list(zip(*list(state_dict_zip.values()))), *list(state_dict.values())):
            _state_space.append(item)
    else:
        for item in product(*list(state_dict.values())):
            _state_space.append(item)
    _state_space_flattened = []
    for entry in _state_space:
        _state_space_flattened.append(tuple(item for element in entry for item in (element if isinstance(element, tuple) else (element,))))
    
    state_space = []
    for _ in range(n_iter):
        state_space.extend(_state_space_flattened)
    for state_value in track(
        state_space,
        description='Running experiment...'
    ):
        run(
            Dictionary, String, Object,
            CaFBECHardwareController, WavemeterLock, MOTMaster,
            lasers, root, wavemeter_info_dirpath, script,
            state_dims, state_value, config_path,
            pre_callback, post_callback, interval,
            **kwargs
        )
    return None


def multi_scan(
    Dictionary,
    String,
    Object,
    CaFBECHardwareController,
    WavemeterLock,
    MOTMaster,
    lasers: List[str],
    root: pathlib.Path,
    wavemeter_info_dirpath: pathlib.Path,
    scripts: List[str],
    motmaster_parameters_with_values: Dict[str, List[Union[int, float]]],
    config_path: str,
    lasers_with_frequencies: Dict[str, List[float]] = None,
    n_iter: int = 1,
    pre_callback: Callable = None,
    post_callback: Callable = None,
    interval: float = 0.01,
    motmaster_parameter_doubles_with_values: Dict[str, List[Union[int, float]]] = None,
    gigatrons: Dict[str, List[float]] = None,
    evalboard: List[Dict[str, Any]] = None,
    **kwargs
) -> None:
    state_dict = {}
    if gigatrons is not None:
        for key, value in gigatrons.items():
            state_dict[f"gigatrons_{key}"] = value
    if lasers_with_frequencies is not None:
        for key, value in lasers_with_frequencies.items():
            state_dict[f"wavemeter_{key}"] = value

    if evalboard is not None:
        _evalboard = []
        for item in evalboard:
            _evalboard.append(str(item))
        state_dict["evalboard"] = _evalboard

    state_dict_zip = {}  
    if motmaster_parameter_doubles_with_values is not None:
        for key, value in motmaster_parameter_doubles_with_values.items():
            state_dict_zip[f"motmaster_{key}"] = value
    for key, value in motmaster_parameters_with_values.items():
        state_dict[f"motmaster_{key}"] = value
    state_dims = list(state_dict_zip.keys()) + list(state_dict.keys())
    _state_space = []
    if motmaster_parameter_doubles_with_values is not None:
        for item in product(list(zip(*list(state_dict_zip.values()))), *list(state_dict.values())):
            _state_space.append(item)
    else:
        for item in product(*list(state_dict.values())):
            _state_space.append(item)
    _state_space_flattened = []
    for entry in _state_space:
        _state_space_flattened.append(tuple(item for element in entry for item in (element if isinstance(element, tuple) else (element,))))
    
    state_space = []
    for _ in range(n_iter):
        state_space.extend(_state_space_flattened)
    for state_value in track(
        state_space,
        description='Running experiment...'
    ):
        for script in scripts:
            run(
                Dictionary, String, Object,
                CaFBECHardwareController, WavemeterLock, MOTMaster,
                lasers, root, wavemeter_info_dirpath, script,
                state_dims, state_value, config_path,
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
        self.config_path = config_path
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
        settings = self.config
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
        if "hardwarecontroller_remote_path_id" in kwargs:
            hardwarecontroller_remote_path_id = kwargs["hardwarecontroller_remote_path_id"]
        else:
            hardwarecontroller_remote_path_id = self.default_remote_path_id
        return motmaster_remote_path_id, wavemeter_remote_path_id, hardwarecontroller_remote_path_id

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
        mmrp_id, wmrp_id, hcrp_id = self.get_remote_path_ids(**kwargs)
        _motmaster = self.__dict__[f"MOTMaster_{mmrp_id}"]
        _wavemeterlock = self.__dict__[f"WavemeterLock_{wmrp_id}"]
        _hardwarecontroller = self.__dict__[f"CaFBECHardwareController_{hcrp_id}"]
        _lasers = self.config["edmsuite_modules"]["WavemeterLock"]["lasers"]
        scan(
            Dictionary, String, Object,
            _hardwarecontroller, _wavemeterlock, _motmaster, _lasers[wmrp_id],
            self.root, self.wavemeter_info_path, script,
            motmaster_parameters_with_values, self.config_path, lasers_with_frequencies,
            n_iter, pre_callback, post_callback, self.interval,
            **kwargs
        )
        return None

    def multi_scan(
        self,
        scripts: List[str],
        motmaster_parameters_with_values: Dict[str, List[Union[int, float]]],
        lasers_with_frequencies: Dict[str, List[float]] = None,
        n_iter: int = 1,
        pre_callback: Callable = None,
        post_callback: Callable = None,
        **kwargs
    ) -> None:
        mmrp_id, wmrp_id, hcrp_id = self.get_remote_path_ids(**kwargs)
        _motmaster = self.__dict__[f"MOTMaster_{mmrp_id}"]
        _wavemeterlock = self.__dict__[f"WavemeterLock_{wmrp_id}"]
        _hardwarecontroller = self.__dict__[f"CaFBECHardwareController_{hcrp_id}"]
        _lasers = self.config["edmsuite_modules"]["WavemeterLock"]["lasers"]
        multi_scan(
            Dictionary, String, Object,
            _hardwarecontroller, _wavemeterlock, _motmaster, _lasers[wmrp_id],
            self.root, self.wavemeter_info_path, scripts,
            motmaster_parameters_with_values, self.config_path, lasers_with_frequencies,
            n_iter, pre_callback, post_callback, self.interval,
            **kwargs
        )
        return None
