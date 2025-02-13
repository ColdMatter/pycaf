from typing import List, Union, Tuple, Dict
import time
import os
import json
from .ad9959 import AD9959


def dds_init_pre_callback(
    state_dims: List[str],
    state_value: List[Union[int, float]],
    **kwargs
) -> None:
    dds_init_motmaster_freq_key: str = kwargs["dds_init_motmaster_freq_key"]
    dds_init_motmaster_amp_key: str = kwargs["dds_init_motmaster_amp_key"]
    exe_path: str = kwargs["dds_init_exe_path"]
    channel: str = str(kwargs["dds_init_channel"])
    if f"motmaster_{dds_init_motmaster_freq_key}" in state_dims:
        freq_index: int = state_dims.index(
            f"motmaster_{dds_init_motmaster_freq_key}"
        )
        frequency = state_value[freq_index]
    else:
        frequency = kwargs["dds_init_default_frequency"]
    if f"motmaster_{dds_init_motmaster_amp_key}" in state_dims:
        amp_index: int = state_dims.index(
            f"motmaster_{dds_init_motmaster_amp_key}"
        )
        amplitude = state_value[amp_index]
    else:
        amplitude = kwargs["dds_init_default_amplitude"]
    os.system(f"{exe_path} F{channel} {frequency} {amplitude}")
    time.sleep(0.1)
    return None


def dds_init_amp_pre_callback(
    **kwargs
) -> None:
    return None


def ad9959_pre_callback(
    state_dims: List[str],
    state_value: List[Union[int, float]],
    **kwargs
) -> None:
    ad9959_instance: AD9959 = kwargs["ad9959_instance"]
    config_index_var: str = kwargs["ad9959_config_index_variable"]
    state_dims_imdex: int = state_dims.index(f"motmaster_{config_index_var}")
    temp_file_path: str = kwargs["ad9959_temp_file_path"]
    config_index: int = state_value[state_dims_imdex]
    channel_configs: Dict[Tuple[float, float]] = \
        kwargs["ad9959_channel_configs"][config_index]
    for channel, config in channel_configs.items():
        ad9959_instance.set_frequency(config[0], channel=channel)
        ad9959_instance.set_amplitude(config[1], channel=channel)
        ad9959_instance.set_phase(0, channel=channel)
    with open(os.path.join(temp_file_path, "AD9959Config.txt"), "w") as f:
        json.dump(channel_configs, f)
    time.sleep(0.1)
    return None
