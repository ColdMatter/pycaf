from typing import List, Dict
from pathlib import Path
from zipfile import ZipFile
import json
import plotly.graph_objects as go

from .analysis import (
    get_zip_archive,
    read_digital_patterns_from_zip,
    read_analog_patterns_from_zip
)
from .analysis import (
    Pattern
)


class PatternVisualizer():
    def __init__(
        self,
        config_path: str,
    ) -> None:
        with open(config_path, "r") as f:
            config = json.load(f)
        self.data_root = Path(config["data_root_path"])
        self.data_prefix = config["data_prefix"]

    def draw(
        self,
        channels: Dict[str, Pattern]
    ) -> None:
        fig = go.Figure()
        for channel, pattern in channels.items():
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    x=pattern.time,
                    y=pattern.event,
                    name=channel
                )
            )
        fig.update_layout(
            xaxis_title="Time [10 us]",
            yaxis_title="Voltage [V]"
        )
        fig.show(
            config={'displaylogo': False}
        )
        return None

    def __call__(
        self,
        years: List[int],
        months: List[int],
        days: List[int],
        file_nos: List[int],
        channel_names: List[str] = None,
        draw: bool = True,
        *args,
        **kwargs
    ) -> Dict[str, Pattern]:
        n_files = len(file_nos)
        if len(years) == 1:
            years: List[int] = [years[0] for _ in range(n_files)]
        if len(months) == 1:
            months: List[int] = [months[0] for _ in range(n_files)]
        if len(days) == 1:
            days: List[int] = [days[0] for _ in range(n_files)]
        channels: Dict[str, Pattern] = {}
        for i, file_no in enumerate(file_nos):
            archive: ZipFile = get_zip_archive(
                self.data_root,
                self.data_prefix,
                years[i],
                months[i],
                days[i],
                file_no
            )
            key = f"{years[i]}_{months[i]}_{days[i]}_{file_no}"
            digital_channels: Dict[str, Pattern] = \
                read_digital_patterns_from_zip(archive)
            analog_channels: Dict[str, Pattern] = \
                read_analog_patterns_from_zip(archive)
            _channels = {
                **digital_channels,
                **analog_channels
            }
            if not channel_names or len(channel_names) == 0:
                channel_names = list(_channels.keys())
            for channel_name in channel_names:
                channels[f"{key}_{channel_name}"] = \
                    _channels.pop(channel_name)
        if draw:
            self.draw(channels)
        return channels
