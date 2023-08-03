from ctypes import byref, c_int16, c_int32, sizeof
from time import sleep
from pathlib import Path

import numpy as np
from picosdk.ps2000 import ps2000
from picosdk.functions import assert_pico2000_ok, adc2mV
from picosdk.PicoDeviceEnums import picoEnum

from pycaf.picoscope.utils import (
    THRESHOLD_DIRECTION,
    TriggerConditions,
    PwqConditions,
    TriggerChannelProperties
)


SAMPLES = 3000
OVERSAMPLING = 1
PRESAMPLE = 10.0
ADC_THRESHOLD = 1000


def get_timebase(device, wanted_time_interval):
    current_timebase = 1

    old_time_interval = None
    time_interval = c_int32(0)
    time_units = c_int16()
    max_samples = c_int32()

    while ps2000.ps2000_get_timebase(
        device.handle,
        current_timebase,
        SAMPLES,
        byref(time_interval),
        byref(time_units),
        1,
        byref(max_samples)) == 0 \
            or time_interval.value < wanted_time_interval:

        current_timebase += 1
        old_time_interval = time_interval.value

        if current_timebase.bit_length() > sizeof(c_int16) * 8:
            raise Exception('No appropriate timebase was identifiable')

    return current_timebase - 1, old_time_interval


def loop(
    datapath: str,
    prefix: str,
    channel_a_range: str,
    channel_b_range: str
) -> None:
    with ps2000.open_unit() as device:
        print(f"Connected Device: {device.info}")
        iteration = 0
        try:
            while True:
                res = ps2000.ps2000_set_channel(
                    device.handle,
                    picoEnum.PICO_CHANNEL['PICO_CHANNEL_A'],
                    True,
                    picoEnum.PICO_COUPLING['PICO_DC'],
                    ps2000.PS2000_VOLTAGE_RANGE[channel_a_range]
                )
                assert_pico2000_ok(res)

                res = ps2000.ps2000_set_channel(
                    device.handle,
                    picoEnum.PICO_CHANNEL['PICO_CHANNEL_B'],
                    True,
                    picoEnum.PICO_COUPLING['PICO_DC'],
                    ps2000.PS2000_VOLTAGE_RANGE[channel_b_range]
                )
                assert_pico2000_ok(res)

                trigger_conditions = TriggerConditions(0, 0, 0, 0, 0, 1)

                res = ps2000.ps2000SetAdvTriggerChannelConditions(
                    device.handle,
                    byref(trigger_conditions),
                    1
                )
                assert_pico2000_ok(res)

                res = ps2000.ps2000SetAdvTriggerDelay(
                    device.handle,
                    0,
                    -PRESAMPLE
                )
                assert_pico2000_ok(res)

                res = ps2000.ps2000SetAdvTriggerChannelDirections(
                    device.handle,
                    THRESHOLD_DIRECTION['PS2000_ADV_RISING_OR_FALLING'],
                    THRESHOLD_DIRECTION['PS2000_ADV_NONE'],
                    THRESHOLD_DIRECTION['PS2000_ADV_NONE'],
                    THRESHOLD_DIRECTION['PS2000_ADV_NONE'],
                    THRESHOLD_DIRECTION['PS2000_ADV_NONE']
                )
                assert_pico2000_ok(res)

                trigger_properties = TriggerChannelProperties(
                    ADC_THRESHOLD,
                    -ADC_THRESHOLD,
                    0,
                    ps2000.PS2000_CHANNEL['PS2000_CHANNEL_A'],
                    picoEnum.PICO_THRESHOLD_MODE['PICO_WINDOW']
                )

                res = ps2000.ps2000SetAdvTriggerChannelProperties(
                    device.handle,
                    byref(trigger_properties),
                    1,
                    0
                )
                assert_pico2000_ok(res)

                pwq_conditions = PwqConditions(1, 0, 0, 0, 0)

                res = ps2000.ps2000SetPulseWidthQualifier(
                    device.handle,
                    byref(pwq_conditions),
                    1,
                    THRESHOLD_DIRECTION['PS2000_ENTER'],
                    200,
                    0,
                    picoEnum.PICO_PULSE_WIDTH_TYPE['PICO_PW_TYPE_GREATER_THAN']
                )
                assert_pico2000_ok(res)

                # calculate timebase
                timebase_a, interval = get_timebase(device, 20_000)

                collection_time = c_int32()

                res = ps2000.ps2000_run_block(
                    device.handle,
                    SAMPLES,
                    timebase_a,
                    OVERSAMPLING,
                    byref(collection_time)
                )
                assert_pico2000_ok(res)

                # wait for ready signal
                while ps2000.ps2000_ready(device.handle) == 0:
                    sleep(0.1)

                times = (c_int32 * SAMPLES)()

                buffer_a = (c_int16 * SAMPLES)()
                buffer_b = (c_int16 * SAMPLES)()

                res = ps2000.ps2000_get_times_and_values(
                    device.handle,
                    byref(times),
                    byref(buffer_a),
                    byref(buffer_b),
                    None,
                    None,
                    None,
                    2,
                    SAMPLES,
                )
                assert_pico2000_ok(res)

                times = np.array(times)
                channel_a_mv = np.array(
                    adc2mV(
                        buffer_a,
                        ps2000.PS2000_VOLTAGE_RANGE[channel_a_range],
                        c_int16(32767)
                    )
                )
                channel_b_mv = np.array(
                    adc2mV(
                        buffer_b,
                        ps2000.PS2000_VOLTAGE_RANGE[channel_b_range],
                        c_int16(32767)
                    )
                )
                channel_a_path = \
                    Path(datapath).joinpath(
                        f"{prefix}_channel_a_{iteration}.txt"
                    )
                channel_b_path = \
                    Path(datapath).joinpath(
                        f"{prefix}_channel_b_{iteration}.txt"
                    )
                np.savetxt(channel_a_path, np.array([times, channel_a_mv]))
                np.savetxt(channel_b_path, np.array([times, channel_b_mv]))
                iteration += 1
        except KeyboardInterrupt:
            print('Stopping picoscope....')
            ps2000.ps2000_stop(device.handle)
    return None


if __name__ == "__main__":
    datapath: str = "C:\\Users\\cafmot\\Documents\\ToF_Data"
    channel_a_range: str = "PS2000_10V"
    channel_b_range: str = "PS2000_1V"
    prefix: str = "tof"
    loop(datapath, prefix, channel_a_range, channel_b_range)
