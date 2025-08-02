from typing import Dict, Any, Callable, List, Tuple, Union
import pathlib
import time
from pylablib.devices import Newport
from rich.progress import track


class PicoMotor8742():
    def __init__(
        self,
        config: Dict[str, Any],
        interval: float,
        cs_Dictionary,
        cs_String,
        cs_Object
    ) -> None:
        self.config = config
        self.interval = interval
        self.Dictionary = cs_Dictionary
        self.String = cs_String
        self.Object = cs_Object
        self.picomotor = self.config["plugin_modules"]["picomotor"]
        self.root = pathlib.Path(self.config["script_root_path"])

    def connect(
        self
    ) -> None:
        if self.picomotor["connect"]:
            self.picomotor_default_axis = None
            self.picomotor_default_speed = None
            self.picomotor_default_acceleration = None
            self.picomotor_default_steps = None
            self.picomotor_default_max_steps = None
            self.picomotor_steps_moved = 0
            try:
                n = Newport.get_usb_devices_number_picomotor()
                if n == 1:
                    self.stage = Newport.Picomotor8742()
                    if "motor" in self.picomotor["defaults"]:
                        self.picomotor_default_motor = \
                            self.picomotor["defaults"]["motor"]
                    if "speed" in self.picomotor["defaults"]:
                        self.picomotor_default_speed = \
                            self.picomotor["defaults"]["speed"]
                    if "aceeleration" in self.picomotor["defaults"]:
                        self.picomotor_default_acceleration = \
                            self.picomotor["defaults"]["acceleration"]
                    if "steps" in self.picomotor["defaults"]:
                        self.picomotor_default_steps = \
                            self.picomotor["defaults"]["steps"]
                    if "max_steps" in self.picomotor["defaults"]:
                        self.picomotor_default_max_steps = \
                            self.picomotor["defaults"]["max_steps"]
                elif n == 0:
                    print("No PicoMotor device detected!")
                else:
                    print("Too many PicoMotor device detected!")
            except Exception as e:
                print(f"Error: {e} encountered")
        return None

    def move_picomotor_with_default_settings(
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
                axis=self.picomotor["motor_to_axis"][motor],
                steps=self.picomotor_default_steps
            )
            self.stage.wait_move()
            self.picomotor_steps_moved += abs(self.picomotor_default_steps)
            if self.picomotor_steps_moved >= \
                    self.picomotor_default_max_steps:
                self.picomotor_default_steps *= -1.0
        return None

    def scan_picomotor_steps(
        self,
        MOTMaster,
        script: str,
        motor: str,
        interval_steps: int,
        total_steps: int,
        speed: int = None,
        accel: int = None,
        pre_callback: Callable = None,
        post_callback: Callable = None,
        motmaster_parameter: str = None,
        motmaster_values: List[Tuple[Union[int, float]]] = None
    ) -> None:
        _dictionary = self.Dictionary[self.String, self.Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        try:
            MOTMaster.SetScriptPath(path)
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
                        if pre_callback is not None:
                            pre_callback()
                        MOTMaster.Go(_dictionary)
                        if post_callback is not None:
                            post_callback()
                        time.sleep(self.interval)
                else:
                    MOTMaster.Go()
                    if post_callback is not None:
                        post_callback()
                    time.sleep(self.interval)
        except Exception as e:
            print(f"Error: {e} encountered")
        return None
