from typing import Any, Callable, Dict, List, Tuple, Union
import pathlib
import json
import sys

# NOTE: these imports will only work with the pythonnet package
try:
    import clr
    from System.Collections.Generic import Dictionary
    from System import String, Object
    from System import Activator
except Exception as e:
    print(f"Error: {e} encountered, probably no pythonnet")


from pycaf.modules import (
    single_run,
    scan_parameter,
    scan_parameters,
    get_laser_set_points,
    scan_laser_set_points,
    scan_laser_set_points_with_motmaster_values,
    scan_laser_set_points_with_motmaster_multiple_parameters,
    PicoMotor8742
)


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

    def get_remote_path_id(
        self,
        **kwargs
    ) -> str:
        if "remote_path_id" in kwargs:
            remote_path_id = kwargs["remote_path_id"]
        else:
            remote_path_id = self.default_remote_path_id
        return remote_path_id

    def motmaster_single_run(
        self,
        script: str,
        parameter: str,
        value: Union[int, float],
        pre_callback: Callable = None,
        post_callback: Callable = None,
        **kwargs
    ) -> None:
        remote_path_id = self.get_remote_path_id(**kwargs)
        _motmaster = self.__dict__[f"MOTMaster_{remote_path_id}"]
        single_run(
            Dictionary, String, Object,
            _motmaster, self.root, script,
            parameter, value, pre_callback, post_callback,
            self.interval
        )
        return None

    def motmaster_scan_parameter(
        self,
        script: str,
        parameter: str,
        values: List[Union[int, float]],
        pre_callback: Callable = None,
        post_callback: Callable = None,
        **kwargs
    ) -> None:
        remote_path_id = self.get_remote_path_id(**kwargs)
        _motmaster = self.__dict__[f"MOTMaster_{remote_path_id}"]
        scan_parameter(
            Dictionary, String, Object,
            _motmaster, self.root, script,
            parameter, values, pre_callback, post_callback,
            self.interval
        )
        self.wavemeter_get_laser_set_points()
        return None

    def motmaster_scan_parameters(
        self,
        script: str,
        parameters: List[str],
        values: List[Tuple[Any]],
        pre_callback: Callable = None,
        post_callback: Callable = None,
        **kwargs
    ) -> None:
        remote_path_id = self.get_remote_path_id(**kwargs)
        _motmaster = self.__dict__[f"MOTMaster_{remote_path_id}"]
        scan_parameters(
            Dictionary, String, Object,
            _motmaster, self.root, script,
            parameters, values, pre_callback, post_callback,
            self.interval
        )
        return None

    def wavemeter_get_laser_set_points(
        self,
        **kwargs
    ) -> Dict[str, float]:
        remote_path_id = self.get_remote_path_id(**kwargs)
        _wavemeterlock = self.__dict__[f"WavemeterLock_{remote_path_id}"]
        _lasers = self.config["edmsuite_modules"]["WavemeterLock"]["lasers"]
        get_laser_set_points(
            _wavemeterlock,
            _lasers[remote_path_id]
        )
        return None

    def wavemeter_scan_laser_set_points(
        self,
        script: str,
        laser: str,
        values: List[Tuple[Any]],
        motmaster_parameter: str = None,
        motmaster_value: Union[int, float] = None,
        pre_callback: Callable = None,
        post_callback: Callable = None,
        **kwargs
    ) -> None:
        remote_path_id = self.get_remote_path_id(**kwargs)
        _motmaster = self.__dict__[f"MOTMaster_{remote_path_id}"]
        _wavemeterlock = self.__dict__[f"WavemeterLock_{remote_path_id}"]
        scan_laser_set_points(
            Dictionary, String, Object,
            _wavemeterlock, _motmaster,
            self.root, script, laser, values,
            pre_callback, post_callback,
            motmaster_parameter, motmaster_value,
            self.interval
        )
        return None

    def wavemeter_scan_laser_set_points_with_motmaster_values(
        self,
        script: str,
        laser: str,
        values: List[Tuple[Any]],
        motmaster_parameter: str = None,
        motmaster_values: List[Union[int, float]] = None,
        pre_callback: Callable = None,
        post_callback: Callable = None,
        **kwargs
    ) -> None:
        remote_path_id = self.get_remote_path_id(**kwargs)
        _motmaster = self.__dict__[f"MOTMaster_{remote_path_id}"]
        _wavemeterlock = self.__dict__[f"WavemeterLock_{remote_path_id}"]
        scan_laser_set_points_with_motmaster_values(
            Dictionary, String, Object,
            _wavemeterlock, _motmaster,
            self.root, script, laser, values,
            pre_callback, post_callback,
            motmaster_parameter, motmaster_values,
            self.interval
        )
        return None

    def wavemeter_scan_laser_set_points_with_motmaster_multiple_parameters(
        self,
        script: str,
        laser: str,
        values: List[Tuple[Any]],
        motmaster_parameters: List[str] = None,
        motmaster_values: List[Tuple[Any]] = None,
        pre_callback: Callable = None,
        post_callback: Callable = None,
        **kwargs
    ) -> None:
        remote_path_id = self.get_remote_path_id(**kwargs)
        _motmaster = self.__dict__[f"MOTMaster_{remote_path_id}"]
        _wavemeterlock = self.__dict__[f"WavemeterLock_{remote_path_id}"]
        scan_laser_set_points_with_motmaster_multiple_parameters(
            Dictionary, String, Object,
            _wavemeterlock, _motmaster,
            self.root, script, laser, values,
            pre_callback, post_callback,
            motmaster_parameters, motmaster_values,
            self.interval
        )
        return None
