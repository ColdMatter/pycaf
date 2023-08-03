from typing import List
import json
from pathlib import Path


class Analysis():
    def __init__(
        self,
        config_path: str,
        year: int,
        month: int,
        day: int
    ) -> None:
        with open(config_path, "r") as f:
            config = json.load(f)
        self.data_root = Path(config["data_root_path"])
        self.data_prefix = config["data_prefix"]
        self.year = year
        self.month = month
        self.day = day

    def __call__(
        self,
        file_nos: List[int],
    ) -> None:
        return None
