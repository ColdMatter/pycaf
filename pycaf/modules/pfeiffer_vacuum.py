from typing import List
import serial
import json


# ------- Control Symbols as defined on p. 81 of the english
#        manual for the Pfeiffer Vacuum TPG256A  -----------
C = {
  'ETX': "\x03",  # End of Text (Ctrl-C)   Reset the interface
  'CR':  "\x0D",  # Carriage Return        Go to the beginning of line
  'LF':  "\x0A",  # Line Feed              Advance by one line
  'ENQ': "\x05",  # Enquiry                Request for data transmission
  'ACQ': "\x06",  # Acknowledge            Positive report signal
  'NAK': "\x15",  # Negative Acknowledge   Negative report signal
  'ESC': "\x1b",  # Escape
}

LINE_TERMINATION = C['CR']+C['LF']  # CR, LF and CRLF are all possible (p.82)


# Mnemonics as defined on p. 85
M = [
  'BAU',  # Baud rate                           95
  'CAx',  # Calibration factor Sensor x         92
  'CID',  # Measurement point names             88
  'DCB',  # Display control Bargraph            89
  'DCC',  # Display control Contrast            90
  'DCD',  # Display control Digits              88
  'DCS',  # Display control Screensave          90
  'DGS',  # Degas                               93
  'ERR',  # Error Status                        97
  'FIL',  # Filter time constant                92
  'FSR',  # Full scale range of linear sensors  93
  'LOC',  # Parameter setup lock                91
  'NAD',  # Node (device) address for RS485     96
  'OFC',  # Offset correction                   93
  'OFC',  # Offset correction                   93
  'PNR',  # Program number                      98
  'PRx',  # Status, Pressure sensor x (1 ... 6) 88
  'PUC',  # Underrange Ctrl                     91
  'RSX',  # Interface                           94
  'SAV',  # Save default                        94
  'SCx',  # Sensor control                      87
  'SEN',  # Sensor on/off                       86
  'SPx',  # Set Point Control Source for Relay x90
  'SPS',  # Set Point Status A,B,C,D,E,F        91
  'TAI',  # Test program A/D Identify           100
  'TAS',  # Test program A/D Sensor             100
  'TDI',  # Display test                        98
  'TEE',  # EEPROM test                         100
  'TEP',  # EPROM test                          99
  'TID',  # Sensor identification               101
  'TKB',  # Keyboard test                       99
  'TRA',  # RAM test                            99
  'UNI',  # Unit of measurement (Display)       89
  'WDT',  # Watchdog and System Error Control   101
]


# Error codes as defined on p. 97
ERR_CODES = [
  {
        0: 'No error',
        1: 'Watchdog has responded',
        2: 'Task fail error',
        4: 'IDCX idle error',
        8: 'Stack overflow error',
        16: 'EPROM error',
        32: 'RAM error',
        64: 'EEPROM error',
        128: 'Key error',
        4096: 'Syntax error',
        8192: 'Inadmissible parameter',
        16384: 'No hardware',
        32768: 'Fatal error'
  },
  {
        0: 'No error',
        1: 'Sensor 1: Measurement error',
        2: 'Sensor 2: Measurement error',
        4: 'Sensor 3: Measurement error',
        8: 'Sensor 4: Measurement error',
        16: 'Sensor 5: Measurement error',
        32: 'Sensor 6: Measurement error',
        512: 'Sensor 1: Identification error',
        1024: 'Sensor 2: Identification error',
        2048: 'Sensor 3: Identification error',
        4096: 'Sensor 4: Identification error',
        8192: 'Sensor 5: Identification error',
        16384: 'Sensor 6: Identification error',
  }
]

# pressure status as defined on p.88
PRESSURE_READING_STATUS = {
  0: 'Measurement data okay',
  1: 'Underrange',
  2: 'Overrange',
  3: 'Sensor error',
  4: 'Sensor off',
  5: 'No sensor',
  6: 'Identification error'
}


class MaxiGauge(object):
    def __init__(
        self,
        config_path: str,
        debug: bool = False
    ) -> None:
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.debug = debug
        serial_port = self.config["plugin_modules"]["pfeiffer"]["serial_port"]
        baud_rate = self.config["plugin_modules"]["pfeiffer"]["baud_rate"]
        try:
            self.connection = serial.Serial(
                serial_port,
                baudrate=baud_rate,
                timeout=0.2
            )
        except serial.SerialException as se:
            raise MaxiGaugeError(se)

    def checkDevice(
        self
    ) -> str:
        message = "The Display Contrast is currently set to " + \
            f"{self.displayContrast()} (out of 20).\n"
        message += "Keys since MaxiGauge was switched on: " + \
            "%s (out of 1,2,3,4,5).\n" % ", ".join(
                map(str, self.pressedKeys())
            )
        return message

    def pressedKeys(
        self
    ) -> List[int]:
        keys = int(self.send('TKB', 1)[0])
        pressedKeys = []
        for i in [4, 3, 2, 1, 0]:
            if keys/2**i == 1:
                pressedKeys.append(i+1)
                keys = keys % 2**i
        pressedKeys.reverse()
        return pressedKeys

    def displayContrast(
        self,
        newContrast: int = -1
    ) -> int:
        if newContrast == -1:
            return int(self.send('DCC', 1)[0])
        else:
            return int(self.send('DCC,%d' % (newContrast, ), 1)[0])

    def pressures(
        self
    ) -> List[float]:
        return [self.pressure(i+1) for i in range(6)]

    def pressure(
        self,
        sensor: int
    ):
        if sensor < 1 or sensor > 6:
            raise MaxiGaugeError(
                'Sensor can only be between 1 and 6. You choose ' + str(sensor)
            )
        reading = self.send('PR%d' % sensor, 1)
        try:
            r = reading[0].split(',')
            status = int(r[0])
            pressure = float(r[-1])
        except Exception as e:
            raise MaxiGaugeError(
                f"Problem interpreting the returned line:\n {reading} with {e}"
            )
        return PressureReading(sensor, status, pressure)

    def debugMessage(
        self,
        message
    ) -> None:
        if self.debug:
            print(repr(message))

    def send(
        self,
        mnemonic: str,
        numEnquiries: int = 0
    ) -> List:
        self.connection.flushInput()
        self.write(mnemonic+LINE_TERMINATION)
        self.getACQorNAK()
        response = []
        for i in range(numEnquiries):
            self.enquire()
            response.append(self.read())
        return response

    def write(
        self,
        what: str
    ) -> None:
        self.debugMessage(what)
        self.connection.write(what.encode())

    def enquire(self) -> None:
        self.write(C['ENQ'])

    def read(self):
        data = ""
        while True:
            x = self.connection.read()
            x = x.decode()
            self.debugMessage(x)
            data += x
            if len(data) > 1 and data[-2:] == LINE_TERMINATION:
                break
        return data[:-len(LINE_TERMINATION)]

    def getACQorNAK(self):
        returncode = self.connection.readline()
        self.debugMessage(returncode)
        # The following is usually expected but our MaxiGauge controller
        # sometimes forgets this parameter... That seems to be a bug with
        # the DCC command.
        # if len(returncode)<3:
        #   raise MaxiGaugeError(
        #       'Only received a line termination from MaxiGauge.' +
        #       'Was expecting ACQ or NAK.'
        # )
        if len(returncode) < 3:
            self.debugMessage(
                'Only received a line termination from MaxiGauge.' +
                'Was expecting ACQ or NAK.'
            )
        if len(returncode) > 2 and returncode[-3] == C['NAK']:
            self.enquire()
            returnedError = self.read()
            error = str(returnedError).split(',', 1)
            print(repr(error))
            errmsg = {
                'System Error': ERR_CODES[0][int(error[0])],
                'Gauge Error': ERR_CODES[1][int(error[1])]
            }
            raise MaxiGaugeNAK(errmsg)
        if len(returncode) > 2 and returncode[-3] != C['ACQ']:
            self.debugMessage(
                'Expecting ACQ or NAK from MaxiGauge but neither were sent.'
            )
        # if no exception raised so far, the interface is just fine:
        return returncode[:-(len(LINE_TERMINATION)+1)]

    def disconnect(self) -> None:
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()

    def __del__(self) -> None:
        self.disconnect()


class PressureReading(object):
    def __init__(
        self,
        id: int,
        status: int,
        pressure: float
    ) -> None:
        if int(id) not in range(1, 7):
            raise MaxiGaugeError('Pressure Gauge ID must be between 1-6')
        self.id = int(id)
        if int(status) not in PRESSURE_READING_STATUS.keys():
            raise MaxiGaugeError(
                "The Pressure Status must be in the range " +
                f"{PRESSURE_READING_STATUS.keys()}"
            )
        self.status = int(status)
        self.pressure = float(pressure)

    def statusMsg(self):
        return PRESSURE_READING_STATUS[self.status]

    def __repr__(self):
        return \
            f"Gauge {self.id}: Status {self.status} ({self.statusMsg()}), " + \
            f"Pressure: {self.pressure} mbar\n"


# ------ now we define the exceptions that could occur ------


class MaxiGaugeError(Exception):
    pass


class MaxiGaugeNAK(MaxiGaugeError):
    pass
