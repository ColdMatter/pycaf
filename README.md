# PyCaF

# Automated experimentation along with data analysis with Python

This repository contains Python code for data analysis tasks using popular libraries like Pandas, NumPy, and Matplotlib. The purpose of this project is to showcase different data analysis techniques and provide examples that can be used as a reference.

## Getting Started

To get started, install anaconda in your system.

Clone this repository
```
git clone https://github.com/ColdMatter/pycaf.git
```

Then create a new environment using the command
```
conda create -f environment.yml
```

After creating the environment activate the environment using
```
conda activate pycaf
```
Navigate to the pycaf directory
```
cd \your_path\pycaf
```
Here install pycaf in editable mode using the command:

```
pip install -e .
```

## Dependencies

The main libraries used in this project are:

- Pythonnet: Used for the python to C# bridge.
- NumPy: Used for numerical computing.
- Matplotlib: Used for creating visualizations.
- Plotly: Used for better interactive visualization.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request. Contributions are welcome and encouraged!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to acknowledge the open-source community for creating and maintaining the essential Python libraries used in this project. Their efforts make data analysis with Python accessible and efficient for everyone.


## Example Configuration file setup for Blackett Bay2 BEC experiment
```json
{
    "edmsuite_modules":
        {
            "MOTMaster":
                {
                    "exe_path": "C:\\ControlPrograms\\EDMSuite\\MOTMaster\\bin\\CaFBEC\\MOTMaster.exe",
                    "remote_paths":
                        {
                            "bay_2": "tcp://localhost:1187/controller.rem",
                            "bay_3": "tcp://172.22.116.195:1187/controller.rem"
                        }
                },
            "WavemeterLock":
                {
                    "exe_path": "C:\\ControlPrograms\\EDMSuite\\WavemeterLock\\bin\\CaFBEC\\WavemeterLock.exe",
                    "remote_paths": 
                        {
                            "bay_2": "tcp://localhost:1234/controller.rem",
                            "bay_3": "tcp://172.22.116.195:5555/controller.rem"
                        },
                    "lasers":
                        {
                            "bay_2": ["v00", "BX", "v10", "Ref"],
                            "bay_3": ["v2, v3"]
                        }
                },
            "CaFBECHardwareController":
                {
                    "exe_path": "C:\\ControlPrograms\\EDMSuite\\CaFBECHardwareController\\bin\\CaFBEC\\CaFBECHardwareController.exe",
                    "remote_paths":
                        {
                            "bay_2": "tcp://localhost:1172/controller.rem"
                        }
                },
            "TransferCavityLock2012":
                {
                    "exe_path": "C:\\ControlPrograms\\EDMSuite\\TransferCavityLock2012\\bin\\CaFBEC\\TransferCavityLock.exe",
                    "remote_paths":
                        {
                            "bay_2": "tcp://localhost:1190/controller.rem"
                        },
                    "lasers": {}
                }
        },
    "edmsuite_dlls":
        {
            "daq": "C:\\ControlPrograms\\EDMSuite\\DAQ\\bin\\CaF\\DAQ.dll",
            "shared_code": "C:\\ControlPrograms\\EDMSuite\\SharedCode\\bin\\CaFBEC\\SharedCode.dll"
        },
    "plugin_modules":
        {
            "picomotor":
                {
                    "type": "pre",
                    "connect": true,
                    "defaults": {
                        "motor": "CMH",
                        "speed": 50,
                        "acceleration": 10,
                        "steps": 10,
                        "max_steps": 300
                    },
                    "motor_to_axis":
                        {
                            "FMH": 1,
                            "FMV": 2,
                            "CMH": 3,
                            "CMV": 4,
                            "Further Mirror Horizontal": 1,
                            "Further Mirror Vertical": 2,
                            "Closer Mirror Horizontal": 3,
                            "Closer Mirror Vertical": 4
                        }
                },
            "evalboard_ad9959": {
                    "board_to_address_map": {"1": "1", "2": "3", "3": "4", "4": "2"},
                    "Board1":
                        {
                            "Channel0": {"frequency": 74.75e6, "amplitude": 0.48, "descr": "V00B1-AOM(BMOT)"},
                            "Channel1": {"frequency": 73.58e6, "amplitude": 0.60, "descr": "V00B1-AOM(LMol)"},
                            "Channel2": {"frequency": 75.5e6, "amplitude": 0.35, "descr": "V00B2AOM"},
                            "Channel3": {"frequency": 76.5e6, "amplitude": 0.65, "descr": "V00R1+AOM(BMOT)"}
                        },
                    "Board2":
                        {
                            "Channel0": {"frequency": 100.0e6, "amplitude": 0.32, "descr": "BXAOM"},
                            "Channel1": {"frequency": 26.0e6, "amplitude": 0.1, "descr": "BX26MHzEOM"},
                            "Channel2": {"frequency": 4.0e6, "amplitude": 0.1, "descr": "BX4MHzEOM"},
                            "Channel3": {"frequency": 76.5e6, "amplitude": 0.79, "descr": "V00R1+(RMOT)"}
                        },
                    "Board3":
                        {
                            "Channel0": {"frequency": 120.0e6, "amplitude": 0.40, "descr": "RepumpAOM"},
                            "Channel1": {"frequency": 100.0e6, "amplitude": 0.50, "descr": "Repump100MHzEOM"},
                            "Channel2": {"frequency": 4.0e6, "amplitude": 0.20, "descr": "Repump4MHzEOM"},
                            "Channel3": {"frequency": 24.0e6, "amplitude": 0.18, "descr": "Repump24MHzEOM"}
                        },
                    "Board4":
                        {
                            "Channel0": {"frequency": 70.7e6, "amplitude": 0.0, "descr": ""},
                            "Channel1": {"frequency": 70.7e6, "amplitude": 0.07, "descr": "V00R0EOM"},
                            "Channel2": {"frequency": 101.0e6, "amplitude": 0.0, "descr": "BXAOM2"},
                            "Channel3": {"frequency": 80.0e6, "amplitude": 0.32, "descr": "ODTAOM"}
                        }
                },
            "picoscope": {},
            "pfeiffer":
                {
                    "serial_port": "COM23",
                    "baud_rate": 9600,
                    "description": {
                        "1": "Source backing",
                        "2": "Science chamber backing",
                        "3": "Not In Use",
                        "4": "Source",
                        "5": "Beamline",
                        "6": "Science chamber"
                    }
                },
            "lakeshore":
                {
                    "serial_port": "COM12",
                    "baud_rate": 57600,
                    "description": {
                        "0": "Cell",
                        "1": "4K",
                        "2": "SF6-4K",
                        "3": "40K",
                        "4": "SF6-40K",
                        "5": "Top Coil",
                        "6": "Jacket",
                        "7": "Cold head"
                    },
                    "sensor-id": {
                        "0": "A",
                        "1": "B",
                        "2": "C",
                        "3": "D1",
                        "4": "D2",
                        "5": "D3",
                        "6": "D4",
                        "7": "D5"
                    }
                }
        },
    "script_root_path": "C:\\ControlPrograms\\EDMSuite\\BECMOTMasterScripts",
    "data_root_path": "C:\\Users\\cafmot\\OneDrive - Imperial College London\\cafbec\\mot_master_data",
    "processed_root_path": "C:\\Users\\cafmot\\OneDrive - Imperial College London\\cafbec\\processed",
    "temp_image_path": "C:\\Users\\cafmot\\Documents\\Temp_camera_images",
    "temp_tof_path": "C:\\Users\\cafmot\\Documents\\ToF_Data",
    "temp_wavemeter_info_path": "C:\\Users\\cafmot\\Documents\\ToF_Data",
    "data_prefix": "CaFBEC",
    "default_remote_path_id": "bay_2",
    "constants":
        {
            "full_well_capacity": 140000.0,
            "bits_per_channel": 16.0,
            "gamma": 2.1e6,
            "collection_solid_angle": 0.0072,
            "eta_q": 0.92,
            "magnification": 0.8,
            "pixel_size": 16e-6,
            "photon_to_electron": 2.2,
            "binning": 4,
            "cs_exposure_time_parameter": "CameraTriggerDuration",
            "cs_camera_trigger_channel_name": "cameraTrigger",
            "mass": 59,
            "v00_set": "494.431890",
            "v10_set": "476.958910",
            "v21_set": "477.299380",
            "v32_set": "477.628175",
            "BX_set": "564.582285"
        }
}
```

### Example api usage

#### Create the experiment and probe object
```python
import numpy as np
import datetime
import pycaf

today = datetime.date.today()
config_path = "C:\\ControlPrograms\\pycaf\\config_bec.json"

expt = pycaf.Experiment(config_path=config_path, interval=0.1)
probe = pycaf.ProbeV3(config_path, today.year, today.month, today.day)
```

#### Run a single experimet
```python
expt.scan(
    script="MOTBasic",
    motmaster_parameters_with_values={
        "yagONorOFF": [10.0, 1.0]
    },
    lasers_with_frequencies={
        "v00": np.arange(494.431850, 494.431880, 0.000003).tolist()
    },
    n_iter=10
)
```

#### Run multiple experimets iteratively
```python
expt.multi_scan(
    scripts=["StackedCompressedMOT", "StackedLambdaMolassesRecapture"],
    motmaster_parameters_with_values={
        "MOTCompressionFieldValue": [3.0+i*0.5 for i in range(11)],
    },
    n_iter=10
)
```

#### Analyse the experimental data
```python
probe(
    115, 204,
    ["MOTCompressionDuration"],
    is_bg_included=True,
    only_number=True,
    row_start=20, row_end=90,
    col_start=20, col_end=100
)\
.exclude_parameters_by_index([0])\
.set_xoffset(0.0)\
.set_xscale(1e-2)\
.set_xlabel("Compression duration [ms]")\
.fit_number_variation(pycaf.analysis.fit_exponential_without_offset)\
.display_variation()\
.display_images()
```