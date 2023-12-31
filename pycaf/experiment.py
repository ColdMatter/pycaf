from typing import Any, Callable, Dict, List, Tuple, Union
import pathlib
from rich.progress import track
import time
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
        self.motmaster = None
        self.hardware_controller = None
        self.transfer_cavity_lock = None
        self.waveplate_controller = None

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
            elif key == "cafbec_hardware_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import CaFBECHadwareController
                    self.hardware_controller = Activator.GetObject(
                        CaFBECHadwareController.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "caf_hardware_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import MoleculeMOTHadwareControl
                    self.hardware_controller = Activator.GetObject(
                        MoleculeMOTHadwareControl.Controller,
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
            elif key == "wavemeter_lock":
                self._add_ref(path_info["exe_path"])
                try:
                    import WavemeterLock
                    self.wavemeter_lock = Activator.GetObject(
                        WavemeterLock.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "picomotor":
                if "connect" in path_info:
                    if path_info["connect"]:
                        self.picomotor_default_axis = None
                        self.picomotor_default_speed = None
                        self.picomotor_default_acceleration = None
                        self.picomotor_default_steps = None
                        self.picomotor_default_max_steps = None
                        self.picomotor_steps_moved = 0
                        try:
                            from pylablib.devices import Newport
                            n = Newport.get_usb_devices_number_picomotor()
                            if n == 1:
                                self.stage = Newport.Picomotor8742()
                                if "motor" in path_info["defaults"]:
                                    self.picomotor_default_motor = \
                                        path_info["defaults"]["motor"]
                                if "speed" in path_info["defaults"]:
                                    self.picomotor_default_speed = \
                                        path_info["defaults"]["speed"]
                                if "aceeleration" in path_info["defaults"]:
                                    self.picomotor_default_acceleration = \
                                        path_info["defaults"]["acceleration"]
                                if "steps" in path_info["defaults"]:
                                    self.picomotor_default_steps = \
                                        path_info["defaults"]["steps"]
                                if "max_steps" in path_info["defaults"]:
                                    self.picomotor_default_max_steps = \
                                        path_info["defaults"]["max_steps"]
                            elif n == 0:
                                print("No PicoMotor device detected!")
                            else:
                                print("Too many PicoMotor device detected!")
                        except Exception as e:
                            print(f"Error: {e} encountered")
        return None

    def disconnect(
        self
    ) -> None:
        self.stage.close()
        return None

    def _move_picomotor_with_default_settings(
        self
    ) -> None:
        if (self.picomotor_default_speed is not None) and \
                (self.picomotor_default_acceleration is not None):
            self.stage.setup_velocity(
                speed=self.picomotor_default_speed,
                accel=self.picomotor_default_acceleration
            )
        if (self.picomotor_default_axis is not None) and \
                (self.picomotor_default_steps is not None):
            motor: str = self.picomotor_default_motor
            self.stage.move_by(
                axis=self.config["picomotor"]["motor_to_axis"][motor],
                steps=self.picomotor_default_steps
            )
            self.stage.wait_move()
            self.picomotor_steps_moved += abs(self.picomotor_default_steps)
            if self.picomotor_steps_moved >= \
                    self.picomotor_default_max_steps:
                self.picomotor_default_steps *= -1.0
        return None

    def get_laser_set_points_tcl(
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

    def get_laser_set_points_wml(
        self
    ) -> Dict[str, Dict[str, float]]:
        lasers = {}
        for laser, _ in self.config["lasers"].items():
            set_point = self.wavemeter_lock.getSlaveFrequency(
                laser
            )
            lasers[laser] = {"set_point": set_point}
            print(
                f"{laser}: set_point = {set_point}"
            )
        return lasers

    def scan_motmaster_parameter(
        self,
        script: str,
        parameter: str,
        values: List[Union[int, float]],
        move_yag_spot: bool = False,
        callback: Callable = None
    ) -> List[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            for i in track(range(len(values))):
                _dictionary[parameter] = values[i]
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                self.motmaster.Go(_dictionary)
                time.sleep(self.interval)
                if callback is not None:
                    result = callback(values[i])
                    results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results

    def scan_motmaster_parameters(
        self,
        script: str,
        parameters: List[str],
        values: List[Tuple[Any]],
        move_yag_spot: bool = False,
        callback: Callable = None
    ) -> List[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            for i in track(range(len(values))):
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                for k, parameter in enumerate(parameters):
                    _dictionary[parameter] = values[i][k]
                self.motmaster.Go(_dictionary)
                time.sleep(self.interval)
                if callback is not None:
                    result = callback(values[i])
                    results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results

    def scan_tcl_laser_set_points(
        self,
        script: str,
        laser: str,
        values: List[Union[int, float]],
        move_yag_spot: bool = False,
        callback: Callable = None,
        motmaster_parameter: str = None,
        motmaster_value: Union[int, float] = None
    ) -> List[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        cavity = self.config["lasers"][laser]
        current_set_point = self.transfer_cavity_lock.GetLaserSetpoint(
            cavity, laser
        )
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            if move_yag_spot:
                self._move_picomotor_with_default_settings()
            if (motmaster_parameter and motmaster_value) is not None:
                _dictionary[motmaster_parameter] = motmaster_value
            while current_set_point > values[0]:
                current_set_point -= 0.001
                self.transfer_cavity_lock.SetLaserSetpoint(
                    cavity, laser, current_set_point
                )
                time.sleep(0.05)
            while current_set_point < values[0]:
                current_set_point += 0.001
                self.transfer_cavity_lock.SetLaserSetpoint(
                    cavity, laser, current_set_point
                )
                time.sleep(0.1)
            for i in track(range(len(values))):
                self.transfer_cavity_lock.SetLaserSetpoint(
                    cavity, laser, values[i]
                )
                self.motmaster.Go(_dictionary)
                time.sleep(self.interval)
                if callback is not None:
                    result = callback(values[i])
                    results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results

    def scan_wm_laser_set_points(
        self,
        script: str,
        laser: str,
        values: List[Union[int, float]],
        move_yag_spot: bool = False,
        callback: Callable = None,
        motmaster_parameter: str = None,
        motmaster_value: Union[int, float] = None
    ) -> List[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        current_set_point = float(self.wavemeter_lock.getSlaveFrequency(
            laser
        ))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            if move_yag_spot:
                self._move_picomotor_with_default_settings()
            if (motmaster_parameter and motmaster_value) is not None:
                _dictionary[motmaster_parameter] = motmaster_value
            while current_set_point > values[0]:
                current_set_point -= 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.05)
            while current_set_point < values[0]:
                current_set_point += 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.1)
            for i in track(range(len(values))):
                self.wavemeter_lock.setSlaveFrequency(
                    laser, values[i]
                )
                self.motmaster.Go(_dictionary)
                time.sleep(self.interval)
                if callback is not None:
                    result = callback(values[i])
                    results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results

    def scan_wm_laser_set_points_with_motmaster_values(
        self,
        script: str,
        laser: str,
        values: List[Union[int, float]],
        move_yag_spot: bool = False,
        callback: Callable = None,
        motmaster_parameter: str = None,
        motmaster_values: List[Union[int, float]] = None
    ) -> List[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        current_set_point = float(self.wavemeter_lock.getSlaveFrequency(
            laser
        ))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            while current_set_point > values[0]:
                current_set_point -= 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.05)
            while current_set_point < values[0]:
                current_set_point += 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.1)
            for i in track(range(len(values))):
                self.wavemeter_lock.setSlaveFrequency(
                    laser, values[i]
                )
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                if (motmaster_parameter and motmaster_values) is not None:
                    for k in range(len(motmaster_values)):
                        _dictionary[motmaster_parameter] = motmaster_values[k]
                        self.motmaster.Go(_dictionary)
                        time.sleep(self.interval)
                        if callback is not None:
                            result = callback(values[i])
                            results.append(result)
                else:
                    self.motmaster.Go(_dictionary)
                    time.sleep(self.interval)
                    if callback is not None:
                        result = callback(values[i])
                        results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results

    def scan_wm_laser_set_points_with_motmaster_multiple_parameters(
        self,
        script: str,
        laser: str,
        values: List[Union[int, float]],
        move_yag_spot: bool = False,
        callback: Callable = None,
        motmaster_parameters: List[str] = None,
        motmaster_values: List[Tuple[Union[int, float]]] = None
    ) -> List[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        current_set_point = float(self.wavemeter_lock.getSlaveFrequency(
            laser
        ))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            while current_set_point > values[0]:
                current_set_point -= 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.05)
            while current_set_point < values[0]:
                current_set_point += 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.1)
            for i in track(range(len(values))):
                self.wavemeter_lock.setSlaveFrequency(
                    laser, values[i]
                )
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                if (motmaster_parameters and motmaster_values) is not None:
                    for k in range(len(motmaster_values)):
                        motmaster_value: Tuple = motmaster_values[k]
                        for t, parameter in enumerate(motmaster_parameters):
                            _dictionary[parameter] = motmaster_value[t]
                        self.motmaster.Go(_dictionary)
                        time.sleep(self.interval)
                        if callback is not None:
                            result = callback(values[i])
                            results.append(result)
                else:
                    self.motmaster.Go(_dictionary)
                    time.sleep(self.interval)
                    if callback is not None:
                        result = callback(values[i])
                        results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results

    def motmaster_single_run(
        self,
        script: str,
        parameter: str = "",
        value: Union[int, float] = None,
    ) -> None:
        _dictionary = Dictionary[String, Object]()
        if len(parameter):
            _dictionary[parameter] = value
        path = str(self.root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(path)
            self.motmaster.Go(_dictionary)
            time.sleep(self.interval)
        except Exception as e:
            print(f"Error: {e} encountered")
        return None

    def scan_microwave_amplitude(
        self,
        script: str,
        synthesizer: str = "Gigatronics Synthesizer 2",
        values: List[float] = [],
        move_yag_spot: bool = False
    ) -> None:
        path = str(self.root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(path)
            for i in track(range(len(values))):
                self.hardware_controller.tabs[synthesizer].SetAmplitude(
                    values[i]
                )
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                self.motmaster.Go()
                time.sleep(self.interval)
        except Exception as e:
            print(f"Error: {e} encountered")
        return None

    def scan_microwave_frequency(
        self,
        script: str,
        synthesizer: str = "Gigatronics Synthesizer 2",
        values: List[float] = [],
        move_yag_spot: bool = False
    ) -> None:
        path = str(self.root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(path)
            for i in track(range(len(values))):
                self.hardware_controller.tabs[synthesizer].SetFrequency(
                    values[i]
                )
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                self.motmaster.Go()
                time.sleep(self.interval)
        except Exception as e:
            print(f"Error: {e} encountered")
        return None

    def scan_picomotor_steps(
        self,
        script: str,
        motor: str,
        interval_steps: int,
        total_steps: int,
        speed: int = None,
        accel: int = None,
        callback: Callable = None,
        motmaster_parameter: str = None,
        motmaster_values: List[Tuple[Union[int, float]]] = None
    ) -> List[Any]:
        path = str(self.root.joinpath(f"{script}.cs"))
        _dictionary = Dictionary[String, Object]()
        try:
            self.motmaster.SetScriptPath(path)
            results = []
            if (speed and accel) is not None:
                self.stage.setup_velocity(speed=speed, accel=accel)
            axis: int = self.config["picomotor"]["motor_to_axis"][motor]
            n_steps: int = int(total_steps/abs(interval_steps))
            for step_index in track(range(n_steps)):
                self.stage.move_by(axis=axis, steps=interval_steps)
                self.stage.wait_move()
                if (motmaster_parameter and motmaster_values) is not None:
                    for k in range(len(motmaster_values)):
                        _dictionary[motmaster_parameter] = motmaster_values[k]
                        self.motmaster.Go(_dictionary)
                        time.sleep(self.interval)
                        if callback is not None:
                            result = callback(step_index)
                            results.append(result)
                else:
                    self.motmaster.Go()
                    time.sleep(self.interval)
                    if callback is not None:
                        result = callback(step_index)
                        results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results
