import pprint
from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
import nidaqmx
import matplotlib.pyplot as plt

channel = "/PXI1Slot6/ai1"

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(channel)
    task.timing.cfg_samp_clk_timing(1000.0, sample_mode=AcquisitionType.FINITE, samps_per_chan=50)
    data = task.read(READ_ALL_AVAILABLE)
    plt.plot(data)
    plt.show()
