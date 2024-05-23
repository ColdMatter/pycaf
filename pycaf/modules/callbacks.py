from typing import List, Union
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
    freq_index: int = state_dims.index(
        f"motmaster_{dds_init_motmaster_freq_key}"
    )
    frequency = state_value[freq_index]
    amp_index: int = state_dims.index(
        f"motmaster_{dds_init_motmaster_amp_key}"
    )
    amplitude = state_value[amp_index]
    os.system(f"{exe_path} F{channel} {frequency} {amplitude}")
    return None


def dds_init_post_callback(
    **kwargs
) -> None:
    return None
