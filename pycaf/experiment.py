from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pathlib
from tqdm import tqdm
import time
import json
import sys

# NOTE: these imports will only work with the pythonnet package
import clr
from System.Collections.Generic import Dictionary
from System import String, Object
from System import Activator


class Experiment():
    def __init__(
        self,
        root: str,
        config_path: str,
        interval: Union[int, float]
    ) -> None:
        self.root = pathlib.Path(root)
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.interval = interval
        self.motmaster = None
        self.hardware_controller = None
        self.transfer_cavity_lock = None
        self.waveplate_controller = None

    def _add_ref(
        self,
        path: str
    ) -> None:
        _path = pathlib.Path(path)
        sys.path.append(_path.parent[0])
        clr.AddReference(path)
        return None

    def connect(
        self
    ) -> None:
        for path in self.config["dll_paths"].values():
            clr.AddReference(path)
        for key, path_info in self.config.items():
            if key == "dll_paths":
                for path in path_info.values():
                    self._add_ref(path)
            elif key == "motmaster":
                self._add_ref(path_info["exe_path"])
                try:
                    import MOTMaster
                    self.motmaster = Activator.GetObject(
                        MOTMaster.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "hardware_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import MoleculeMOTHardwareControl
                    self.hardware_controller = Activator.GetObject(
                        MoleculeMOTHardwareControl.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "transfer_cavity_lock":
                self._add_ref(path_info["exe_path"])
                try:
                    import TransferCavityLock2012
                    self.transfer_cavity_lock = Activator.GetObject(
                        TransferCavityLock2012.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "wave_plate_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import WavePlateControl
                    self.waveplate_controller = Activator.GetObject(
                        WavePlateControl.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
        return None

    def get_laser_set_points(
        self
    ) -> Dict[str, Dict[str, float]]:
        lasers = {}
        for laser, cavity in self.config["lasers"].items():
            voltage = self.transfer_cavity_lock.GetLaserVoltage(
                cavity, laser
            )
            set_point = self.transfer_cavity_lock.GetLaserSetpoint(
                cavity, laser
            )
            lasers[laser] = {"voltage": voltage, "set_point": set_point}
            print(
                f"{laser}: voltage = {voltage}, set_point = {set_point}"
            )
        return lasers

    def scan_motmaster_parameter(
        self,
        script: str,
        parameter: str,
        values: List[Union[int, float]]
    ) -> None:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(path)
            for i in tqdm(range(len(values))):
                _dictionary[parameter] = values[i]
                self.motmaster.Go(_dictionary)
                time.sleep(self.interval)
        except Exception as e:
            print(f"Error: {e} encountered")
        return None

    def scan_motmaster_parameters(
        self,
        script: str,
        parameters: List[str],
        values: List[Tuple[Any]]
    ) -> None:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(path)
            for i in tqdm(range(len(values))):
                for k, parameter in enumerate(parameters):
                    _dictionary[parameter] = values[i][k]
                self.motmaster.Go(_dictionary)
                time.sleep(self.interval)
        except Exception as e:
            print(f"Error: {e} encountered")
        return None

    def scan_laser_set_points(
        self,
        script: str,
        cavity: str,
        laser: str,
        values: List[Union[int, float]]
    ) -> None:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(path)
            for i in tqdm(range(len(values))):
                self.transfer_cavity_lock.SetLaserSetpoint(
                    cavity, laser, values[i]
                )
                self.motmaster.Go(_dictionary)
                time.sleep(self.interval)
        except Exception as e:
            print(f"Error: {e} encountered")
        return None

    def scan_motmaster_parameters_with_alternating_parameter(
        self,
        script: str,
        parameter: str,
        values: List[Union[int, float]],
        alternating_parameter: str,
        alternating_value: List[Union[int, float]]
    ) -> None:
        return None

    def scan_laser_set_points_with_alternating_parameter(
        self,
        script: str,
        cavity: str,
        laser: str,
        values: List[Union[int, float]],
        alternating_parameter: str,
        alternating_value: List[Union[int, float]]
    ) -> None:
        return None

    def scan_hardware_controller_parameters(
        self,
        script: str,
        parameters: List[str],
        values: List[Union[int, float]] | np.ndarray
    ) -> None:
        return None
