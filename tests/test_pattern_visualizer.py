from pathlib import Path
from pycaf.pattern_visualizer import PatternVisualizer


def test_pattern_visualizer():
    config_path = Path(__file__).\
        parent.absolute().joinpath("test_config.json")
    visualizer = PatternVisualizer(config_path)
    visualizer(
        years=[2022],
        months=[9],
        days=[14],
        file_nos=[0]
    )
