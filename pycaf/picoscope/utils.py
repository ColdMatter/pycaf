from ctypes import c_int16, c_int32, Structure, c_uint16


THRESHOLD_DIRECTION = {
    'PS2000_ABOVE': 0,
    'PS2000_BELOW': 1,
    'PS2000_ADV_RISING': 2,
    'PS2000_ADV_FALLING': 3,
    'PS2000_ADV_RISING_OR_FALLING': 4,
    'PS2000_INSIDE': 0,
    'PS2000_OUTSIDE': 1,
    'PS2000_ENTER': 2,
    'PS2000_EXIT': 3,
    'PS2000_ENTER_OR_EXIT': 4,
    'PS2000_ADV_NONE': 2,
}


class TriggerConditions(Structure):
    _fields_ = [
        ('channelA', c_int32),
        ('channelB', c_int32),
        ('channelC', c_int32),
        ('channelD', c_int32),
        ('external', c_int32),
        ('pulseWidthQualifier', c_int32),
    ]


class PwqConditions(Structure):
    _fields_ = [
        ('channelA', c_int32),
        ('channelB', c_int32),
        ('channelC', c_int32),
        ('channelD', c_int32),
        ('external', c_int32),
    ]


class TriggerChannelProperties(Structure):
    _fields_ = [
        ("thresholdMajor", c_int16),
        ("thresholdMinor", c_int16),
        ("hysteresis", c_uint16),
        ("channel", c_int16),
        ("thresholdMode", c_int16),
    ]
