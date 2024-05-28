from typing import List, Union
import time
import os


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
