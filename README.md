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


## Example Configuration file setup
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
                            "bay_2": ["v00", "BX", "v10", "v21", "v32"],
                            "bay_3": ["TCool"]
                        }
                },
            "CaFBECHadwareController":
                {
                    "exe_path": "C:\\ControlPrograms\\EDMSuite\\CaFBECHadwareController\\bin\\CaFBEC\\CaFBECHadwareController.exe",
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
            "picomotor": {},
            "evalboard_ad9959": {},
            "picoscope": {}
        },
    "script_root_path": "C:\\ControlPrograms\\EDMSuite\\BECMOTMasterScripts",
    "data_root_path": "E:\\mot_master_data",
    "temp_image_path": "C:\\Users\\cafmot\\Documents\\Temp_camera_images",
    "temp_tof_path": "C:\\Users\\cafmot\\Documents\\ToF_Data",
    "data_prefix": "CaFBEC",
    "default_remote_path_id": "bay_2",
    "constants":
        {
            "full_well_capacity": 370000.0,
            "bits_per_channel": 16.0,
            "gamma": 1.5e6,
        }
}
``` 