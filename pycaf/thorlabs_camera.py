"""
Free-running frame grabber and live monitor for Thorlabs scientific cameras
(e.g. CS165MU1/M).

This reproduces the acquisition behaviour that MOTMaster's data pipeline
expects, and which the Hamamatsu/HCImage setup used to provide:

    camera  --(hardware trigger)-->  one single-page 16-bit .tif per frame
            --> ExternalFilesPath ("C:\\Users\\cafmot\\Documents\\TempCameraImages")

MOTMaster does not talk to the camera at all (CameraUsed is false in
MoleculeMOTHardware.cs). At the end of every run it globs "*.tif" out of
ExternalFilesPath, appends them to the run's .zip, and deletes them. So all
this module has to do is keep dropping correctly-named .tif files into that
folder for as long as the experiment is running.

Exposure is set by the trigger level by default (exposure_mode="bulb"): the
camera exposes for as long as the trigger signal is asserted, so the sequence
controls the exposure of every image individually. Set exposure_mode="fixed"
to use a single software exposure time for all frames instead.

Two details matter for correctness and are handled here:

  * Files are written to a ".part" temporary name and then atomically renamed
    to ".tif". MOTMaster can sweep the folder at any instant, so a file must
    never be visible under its final name until it is complete on disk.

  * Frames are single-page TIFFs. ThorCam writes multi-page TIFFs, and the
    analysis readers (analysis3.readFromZip, pycaf read_images_from_zip) use
    PIL's Image.open, which silently returns only the first page.

Usage from a notebook:

    from pycaf.thorlabs_camera import ThorlabsGrabber
    grabber = ThorlabsGrabber(binx=8, biny=8)   # bulb mode by default
    grabber.start()
    monitor = grabber.show_monitor()   # live window, does not block
    ...                                # fire MOTMaster runs as usual
    grabber.stop()

Usage from a terminal (Ctrl-C to stop):

    python -m pycaf.thorlabs_camera --bin 8 --monitor
"""

from typing import Optional, Tuple
from collections import deque
import os
import re
import sys
import glob
import json
import time
import zipfile
import threading

import numpy as np
from PIL import Image


# The native DLLs shipped with ThorCam. They must be on the DLL search path
# before thorlabs_tsi_sdk is imported, otherwise the import fails.
DEFAULT_DLL_DIR = (
    r"C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support"
    r"\Scientific Camera Interfaces\SDK\Native Toolkit\dlls\Native_64_lib"
)

# MOTMaster's ExternalFilesPath on ph-bonesaw, from PHBonesawFileSystem.cs.
DEFAULT_OUTPUT_DIR = r"C:\Users\cafmot\Documents\TempCameraImages"

# Leading "C" keeps these files compatible with the readers that filter CaF
# images on the first character (analysis3.readFromZipCaFRb, diffStr).
DEFAULT_PREFIX = "CImage"

# Camera settings never reach the MOTMaster zip, so they are logged here and
# matched to frames by the counter embedded in each filename.
DEFAULT_SETTINGS_LOG = r"C:\Users\cafmot\Documents\thorlabs_camera_log.jsonl"

# The molecule number calculation is calibrated for one imaging condition:
# the frame taken at Frame0Trigger = 4000 clock periods (40 ms) with
# Frame0TriggerDuration = 1000 (10 ms exposure), the 10 us pattern clock
# giving those numbers. The live plot is shown only when the declared
# sequence matches, because outside it the number would be quietly wrong.
CALIBRATED_FRAME_TRIGGER_MS = 40.0
CALIBRATED_EXPOSURE_MS = 10.0

# Where MOTMaster writes run zips (PHBonesawFileSystem MOTMasterDataPath).
DEFAULT_DATA_ROOT = (
    r"C:\Users\cafmot\Box\Working folder\cafmot\datasync\mot_master_data"
)
DEFAULT_DATA_PREFIX = "CaF"
_MONTHS = {
    1: "01Jan", 2: "02Feb", 3: "03Mar", 4: "04Apr", 5: "05May", 6: "06Jun",
    7: "07Jul", 8: "08Aug", 9: "09Sep", 10: "10Oct", 11: "11Nov", 12: "12Dec",
}


def natural_keys(text: str):
    """Sort 'x_2.tif' before 'x_10.tif', matching the analysis readers."""
    return [
        int(c) if c.isdigit() else c
        for c in re.split(r"(\d+)", text)
    ]


def configure_dll_path(dll_dir: str = None) -> str:
    dll_dir = dll_dir or DEFAULT_DLL_DIR
    if not os.path.isdir(dll_dir):
        raise RuntimeError(
            "Thorlabs native DLL directory not found:\n  {}\n"
            "Is ThorCam installed? Pass dll_dir=... if it lives "
            "elsewhere.".format(dll_dir)
        )
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll_dir)
    return dll_dir


configure_dll_path()

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK  # noqa: E402
from thorlabs_tsi_sdk.tl_camera_enums import (  # noqa: E402
    OPERATION_MODE,
    TRIGGER_POLARITY,
)


class ThorlabsGrabber():
    """Polls a hardware-triggered Thorlabs camera, writes one .tif per frame.

    The grabber arms the camera once and then runs until stop() is called, so
    a whole parameter scan of many MOTMaster files needs no re-arming.

    exposure_mode:
        "bulb"  - the trigger level defines the exposure of each frame. The
                  camera exposes while the trigger is asserted, so the
                  MOTMaster sequence controls exposure image by image.
        "fixed" - every frame uses exposure_us, and the trigger only starts
                  the exposure.
    """

    def __init__(
        self,
        serial_number: Optional[str] = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        prefix: str = DEFAULT_PREFIX,
        exposure_mode: str = "bulb",
        exposure_us: int = 10000,
        binx: int = 8,
        biny: int = 8,
        roi: Optional[Tuple[int, int, int, int]] = None,
        trigger_polarity: str = "active_high",
        frames_to_buffer: int = 32,
        poll_timeout_ms: int = 200,
        clear_stale: bool = True,
        keep_history: int = 200,
        settings_log_path: str = DEFAULT_SETTINGS_LOG,
        crop: Optional[Tuple[int, int, int, int]] = None,
        data_root: str = DEFAULT_DATA_ROOT,
        data_prefix: str = DEFAULT_DATA_PREFIX,
        sequence_frame_trigger_ms: float = CALIBRATED_FRAME_TRIGGER_MS,
        sequence_exposure_ms: float = CALIBRATED_EXPOSURE_MS,
        verbose: bool = True
    ) -> None:
        if str(exposure_mode).lower() not in ("bulb", "fixed"):
            raise ValueError(
                "exposure_mode must be 'bulb' or 'fixed', got "
                "{!r}".format(exposure_mode)
            )
        self.serial_number = serial_number
        self.output_dir = output_dir
        self.prefix = prefix
        self.exposure_mode = str(exposure_mode).lower()
        self.exposure_us = exposure_us
        self.binx = binx
        self.biny = biny
        self.roi = roi
        self.trigger_polarity = trigger_polarity
        self.frames_to_buffer = frames_to_buffer
        self.poll_timeout_ms = poll_timeout_ms
        self.clear_stale = clear_stale
        # consecutive failed polls tolerated before giving up
        self.max_poll_errors = 5
        self.settings_log_path = settings_log_path
        # Software crop, in FULL-SENSOR frame coordinates (r0, r1, c0, c1).
        # Applied before the frame is saved, so the .tif in the zip contains
        # only this region.
        self.crop = tuple(crop) if crop else None
        self.data_root = data_root
        self.data_prefix = data_prefix
        # What the MOTMaster sequence is currently doing. The grabber only
        # sees trigger edges, so this cannot be detected and must be declared.
        self.sequence_frame_trigger_ms = float(sequence_frame_trigger_ms)
        self.sequence_exposure_ms = float(sequence_exposure_ms)
        self.verbose = verbose

        self._sdk = None
        self._camera = None
        self._thread = None
        self._stop_event = threading.Event()
        self._monitor = None

        # Diagnostics, readable at any time while running.
        self.frames_written = 0
        self.frames_dropped = 0
        self.last_error = None
        self.image_width = None
        self.image_height = None
        self.bit_depth = None
        self.operation_mode = None

        # Latest frame, shared with the monitor window. Guarded by a lock
        # because it is written by the polling thread and read by the GUI.
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_index = None
        self._latest_time = None
        self._gain_db = 0.0
        self._background = None
        self._background_geometry = None
        self._background_file_no = None
        self._background_error = None
        self.counts_history = deque(maxlen=keep_history)
        self.net_counts_history = deque(maxlen=keep_history)
        self.number_history = deque(maxlen=keep_history)
        self.recapture_history = deque(maxlen=keep_history)
        self._recapture_buffer = []

        # Writing frames to disk. Turn OFF for MOTMaster "run until stopped":
        # that mode never saves a zip, so MOTMaster never sweeps the folder
        # and the frames pile up and land in the next run that does save.
        # MOTMaster exposes SetRunUntilStopped but no getter over remoting,
        # so this cannot be detected and has to be switched.
        self.saving = True
        # Per-frame derived quantities. Off means not computed at all.
        self.compute_number = True
        self.compute_recapture = False
        # How many camera triggers the sequence fires per shot. Molecule
        # number is computed ONLY on the first frame of each shot: the
        # later frames are a recaptured cloud and an in-sequence
        # background, and including them makes the trace a sawtooth.
        self.frames_per_shot = 1
        self._shot_phase = 0

        self._session = time.strftime("%H%M%S")
        self._counter = 0
        self._last_frame_count = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print("[ThorlabsGrabber] {}".format(message))

    def clear_output_dir(self) -> int:
        """Remove leftover .tif/.part files.

        Stale frames from an aborted run would otherwise be swept into the
        next run's zip and silently corrupt that file's image count.
        """
        removed = 0
        for pattern in ("*.tif", "*.part"):
            for path in glob.glob(os.path.join(self.output_dir, pattern)):
                try:
                    os.remove(path)
                    removed += 1
                except OSError as e:
                    self._log("could not remove {}: {}".format(path, e))
        return removed

    def _open_camera(self) -> None:
        self._sdk = TLCameraSDK()
        available = self._sdk.discover_available_cameras()
        if not available:
            self._sdk.dispose()
            self._sdk = None
            raise RuntimeError(
                "No Thorlabs camera detected. If ThorCam is open, close it "
                "first: it holds the camera exclusively."
            )
        serial = self.serial_number or available[0]
        if serial not in available:
            self._sdk.dispose()
            self._sdk = None
            raise RuntimeError(
                "Camera {} not found. Available: {}".format(serial, available)
            )
        self._camera = self._sdk.open_camera(serial)
        self._log("opened camera {} ({})".format(serial, self._camera.model))

        camera = self._camera
        if self.roi is not None:
            # ROI is in unbinned sensor pixels, so set it before binning.
            camera.roi = self.roi
        camera.binx = self.binx
        camera.biny = self.biny

        # In BULB the trigger signal defines the exposure of each frame, so
        # exposure_time_us is not used and is deliberately left alone.
        mode = (
            OPERATION_MODE.BULB if self.exposure_mode == "bulb"
            else OPERATION_MODE.HARDWARE_TRIGGERED
        )
        if not camera.get_is_operation_mode_supported(mode):
            raise RuntimeError(
                "Camera {} does not support {} mode.".format(
                    camera.model, OPERATION_MODE(mode).name
                )
            )
        if mode == OPERATION_MODE.HARDWARE_TRIGGERED:
            camera.exposure_time_us = self.exposure_us
        camera.operation_mode = mode

        # One frame per trigger, matching the old Hamamatsu behaviour.
        camera.frames_per_trigger_zero_for_unlimited = 1
        low = str(self.trigger_polarity).lower()
        camera.trigger_polarity = (
            TRIGGER_POLARITY.ACTIVE_LOW
            if low in ("active_low", "low", "falling")
            else TRIGGER_POLARITY.ACTIVE_HIGH
        )
        camera.image_poll_timeout_ms = self.poll_timeout_ms

        camera.arm(self.frames_to_buffer)

        self.image_width = camera.image_width_pixels
        self.image_height = camera.image_height_pixels
        self.bit_depth = camera.bit_depth
        self._gain_db = self._read_gain_db()
        self.operation_mode = OPERATION_MODE(camera.operation_mode).name
        exposure_note = (
            "exposure set by trigger level (bulb)"
            if self.exposure_mode == "bulb"
            else "exposure {} us (fixed)".format(camera.exposure_time_us)
        )
        self._log(
            "armed: {}x{} px, {}-bit, bin {}x{}, mode {}, {}, "
            "trigger {}".format(
                self.image_width, self.image_height, self.bit_depth,
                camera.binx, camera.biny, self.operation_mode, exposure_note,
                TRIGGER_POLARITY(camera.trigger_polarity).name
            )
        )

    def _write_frame(self, image: np.ndarray) -> None:
        """Write one frame atomically, so MOTMaster never zips a partial file."""
        name = "{}{}_{:06d}".format(self.prefix, self._session, self._counter)
        final_path = os.path.join(self.output_dir, name + ".tif")
        temp_path = os.path.join(self.output_dir, name + ".part")
        Image.fromarray(image.astype(np.uint16)).save(temp_path, format="TIFF")
        os.replace(temp_path, final_path)
        self._counter += 1
        self.frames_written += 1

    def _read_settings_log(self) -> list:
        entries = []
        try:
            with open(self.settings_log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except ValueError:
                            pass
        except OSError:
            pass
        return entries

    def _geometry_for_frames(self, names) -> Optional[dict]:
        """Crop and binning the given saved frames were taken with.

        Recovered from the settings log by the session and counter embedded
        in each filename. Returns None when the log does not cover them.
        """
        log = self._read_settings_log()
        if not log:
            return None
        pattern = re.compile(r"^[A-Za-z]+(\d+)_(\d+)\.tif$", re.I)
        found = []
        for name in names:
            m = pattern.match(os.path.basename(name))
            if not m:
                continue
            session, counter = m.group(1), int(m.group(2))
            candidates = [
                e for e in log
                if str(e.get("session")) == session
                and int(e.get("from_counter", 0)) <= counter
            ]
            if candidates:
                # Later entries win ties on from_counter: max() alone
                # would return the first, i.e. the superseded setting.
                found.append(max(
                    enumerate(candidates),
                    key=lambda t: (int(t[1]["from_counter"]), t[0]))[1])
        if not found:
            return None
        keys = {
            (tuple(e["crop"]) if e.get("crop") else None,
             e.get("binx"), e.get("biny"), e.get("gain_db"))
            for e in found
        }
        if len(keys) > 1:
            return None  # frames span a change; refuse to guess
        crop, binx, biny, gain_db = keys.pop()
        return {
            "crop": crop, "binx": binx, "biny": biny, "gain_db": gain_db
        }

    def todays_zip_path(self, file_no: int) -> str:
        """Path to one of today's MOTMaster run zips."""
        today = time.localtime()
        return os.path.join(
            self.data_root,
            str(today.tm_year),
            _MONTHS[today.tm_mon],
            "{:02d}".format(today.tm_mday),
            "{}{:02d}{}{}00_{:03d}.zip".format(
                self.data_prefix, today.tm_mday,
                _MONTHS[today.tm_mon][-3:], str(today.tm_year)[-2:],
                int(file_no),
            ),
        )

    def set_background_file(
        self,
        file_no: Optional[int]
    ) -> "ThorlabsGrabber":
        """Use one of today's run zips as the live background.

        Pass None to clear it. Raises with a readable message if the file is
        missing or has no images; the monitor turns that into the same kind
        of notice it shows when the sequence timing is wrong, rather than
        plotting a number that would be quietly wrong.
        """
        if file_no is None:
            with self._frame_lock:
                self._background = None
                self._background_file_no = None
                self._background_error = None
            self._log("background file cleared")
            return self

        path = self.todays_zip_path(file_no)
        if not os.path.isfile(path):
            with self._frame_lock:
                self._background = None
                self._background_file_no = int(file_no)
                self._background_error = "file {:03d} not found today".format(
                    int(file_no)
                )
            raise FileNotFoundError(
                "No run zip {:03d} in today's folder:\n  {}".format(
                    int(file_no), path
                )
            )
        images = []
        with zipfile.ZipFile(path) as z:
            names = sorted(
                (n for n in z.namelist() if n.lower().endswith(".tif")),
                key=natural_keys
            )
            for name in names:
                with z.open(name) as fh:
                    images.append(np.array(Image.open(fh), dtype=float))
        if not images:
            with self._frame_lock:
                self._background = None
                self._background_file_no = int(file_no)
                self._background_error = \
                    "file {:03d} contains no images".format(int(file_no))
            raise ValueError(
                "Run zip {:03d} contains no .tif images".format(int(file_no))
            )
        background = np.mean(np.array(images), axis=0)
        geometry = self._geometry_for_frames(names)
        with self._frame_lock:
            self._background = background
            self._background_geometry = geometry
            self._background_file_no = int(file_no)
            self._background_error = None
        self._log(
            "background from file {:03d}: {} frame(s), shape {}, {}".format(
                int(file_no), len(images), background.shape,
                "crop {} bin {}x{}".format(
                    geometry["crop"], geometry["binx"], geometry["biny"]
                ) if geometry else "geometry unknown (not in settings log)"
            )
        )
        return self

    def _align_background(
        self,
        image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Background matching this frame's region, or a reason it cannot.

        A background taken over a LARGER region than the current crop is
        usable: the settings log records where each region sat on the sensor,
        so the matching sub-region is cut out by absolute pixel position. A
        smaller background cannot cover the current frame, and neither can one
        taken at a different binning, since that changes what a pixel means.
        """
        bg = self._background
        if bg is None:
            return None, "no background file set"
        if bg.shape == image.shape and self._background_geometry is None:
            # Same size and no log information: assume the same region.
            return bg, None

        geometry = self._background_geometry
        if geometry is None:
            return None, (
                "background is {} but frames are {}, and its crop is not in "
                "the settings log - retake the background with this "
                "crop".format(bg.shape, image.shape)
            )
        if (geometry["binx"], geometry["biny"]) != (self.binx, self.biny):
            return None, (
                "background was taken at bin {}x{} but the camera is now at "
                "{}x{} - retake the background".format(
                    geometry["binx"], geometry["biny"], self.binx, self.biny
                )
            )
        bg_gain = geometry.get("gain_db")
        if bg_gain is not None and abs(bg_gain - self._gain_db) > 1e-6:
            return None, (
                "background was taken at {:g} dB but the camera is now at "
                "{:g} dB - counts scale with gain, so retake the "
                "background".format(bg_gain, self._gain_db)
            )

        bg_region = geometry["crop"] or (0, bg.shape[0], 0, bg.shape[1])
        cur_region = self.crop or (0, image.shape[0], 0, image.shape[1])
        br0, br1, bc0, bc1 = bg_region
        cr0, cr1, cc0, cc1 = cur_region

        if (br1 - br0, bc1 - bc0) != bg.shape:
            return None, (
                "background geometry in the log does not match the file - "
                "retake the background"
            )
        if not (br0 <= cr0 and cr1 <= br1 and bc0 <= cc0 and cc1 <= bc1):
            return None, (
                "crop region changed: background covers r{}:{} c{}:{} but "
                "frames need r{}:{} c{}:{} - please update the background "
                "file".format(br0, br1, bc0, bc1, cr0, cr1, cc0, cc1)
            )

        sub = bg[cr0 - br0:cr1 - br0, cc0 - bc0:cc1 - bc0]
        if sub.shape != image.shape:
            return None, (
                "background slice is {} but frames are {} - please update "
                "the background file".format(sub.shape, image.shape)
            )
        return sub, None

    def number_status(self) -> Tuple[bool, str]:
        """Can the live molecule number be trusted, and if not, why not.

        Every reason is reported the same way so a missing trace is never
        ambiguous.
        """
        if not self.is_calibrated_condition:
            return False, (
                "sequence is {:g}/{:g} ms, calibration is {:g}/{:g} ms".format(
                    self.sequence_frame_trigger_ms, self.sequence_exposure_ms,
                    CALIBRATED_FRAME_TRIGGER_MS, CALIBRATED_EXPOSURE_MS
                )
            )
        if self._background_error:
            return False, self._background_error
        frame = self._latest_frame
        if frame is None:
            if self._background is None:
                return False, "no background file set"
            return True, "background file {:03d}".format(
                self._background_file_no
            )
        _, reason = self._align_background(frame)
        if reason:
            return False, reason
        return True, "background file {:03d}".format(self._background_file_no)

    def _background_counts(self, image: np.ndarray) -> Optional[float]:
        """Total background counts to remove, or None if unavailable."""
        bg, _ = self._align_background(image)
        return None if bg is None else float(bg.sum())

    def subtracted(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Frame minus the aligned background, or None if unavailable.

        For display only. Saved .tif files are never background-subtracted.
        """
        bg, _ = self._align_background(image)
        return None if bg is None else image.astype(float) - bg

    @property
    def is_calibrated_condition(self) -> bool:
        """True when the declared sequence matches the molecule-number calibration."""
        return (
            abs(self.sequence_frame_trigger_ms - CALIBRATED_FRAME_TRIGGER_MS) < 1e-6
            and abs(self.sequence_exposure_ms - CALIBRATED_EXPOSURE_MS) < 1e-6
        )

    def _read_gain_db(self) -> float:
        """Gain in dB from the camera, 0 if this model has none."""
        if self._camera is None:
            return 0.0
        try:
            if self._camera.gain_range.max > 0:
                return float(
                    self._camera.convert_gain_to_decibels(self._camera.gain)
                )
        except Exception:
            pass
        return 0.0

    def current_settings(self):
        """ThorlabsSettings describing the camera as it is configured now.

        Uses the cached gain rather than querying the camera, because this is
        called for every frame.
        """
        from .thorlabs_analysis import ThorlabsSettings
        black_level = 0.0
        if self._camera is not None:
            try:
                black_level = float(self._camera.black_level)
            except Exception:
                pass
        return ThorlabsSettings(
            gain_db=self._gain_db,
            binx=self.binx,
            biny=self.biny,
            bit_depth=self.bit_depth or 10,
            exposure_time=self.sequence_exposure_ms * 1e-3,
            black_level=black_level,
        )

    def _publish_frame(self, image: np.ndarray) -> None:
        """Hand the frame to the monitor window."""
        counts = float(image.sum())
        number = np.nan
        net_counts = np.nan
        recapture = None

        # Position of this frame within the shot, before advancing.
        phase = self._shot_phase
        self._shot_phase = (self._shot_phase + 1) % max(
            1, self.frames_per_shot)
        first_of_shot = (phase == 0)

        # Only do the work that something is actually displaying.
        if (self.compute_number and first_of_shot) or self.compute_recapture:
            try:
                background = self._background_counts(image)
            except Exception:
                background = None
            if background is not None:
                net_counts = counts - background
                if (self.compute_number and first_of_shot
                        and self.is_calibrated_condition):
                    try:
                        from .thorlabs_analysis import counts_to_number
                        number = float(counts_to_number(
                            net_counts, self.current_settings()
                        ))
                    except Exception:
                        number = np.nan

        if self.compute_recapture:
            recapture = self._update_recapture(image)

        with self._frame_lock:
            # NOTE: the raw frame is stored, and the raw frame is what
            # _write_frame saves. Background subtraction is display-only.
            self._latest_frame = image
            self._latest_index = self._counter - 1
            self._latest_time = time.time()
            self.counts_history.append(counts)
            self.net_counts_history.append(net_counts)
            if first_of_shot:
                self.number_history.append(number)
            if recapture is not None:
                self.recapture_history.append(recapture)

    def _update_recapture(self, image: np.ndarray) -> Optional[float]:
        """Recapture rate from a triple of frames, or None until one is full.

        Matches get_recapture_rate_three_image: three camera shots per
        sequence, in the order

            0  first image
            1  second image (the recaptured cloud)
            2  in-sequence background, taken later

        so rate = sum(second - bg) / sum(first - bg).

        Frames are grouped in threes from whenever recapture was switched on,
        so if the phase is wrong use reset_recapture() between sequences.
        """
        self._recapture_buffer.append(np.asarray(image, dtype=float))
        if len(self._recapture_buffer) < 3:
            return None
        first, second, bg = self._recapture_buffer
        self._recapture_buffer = []
        if first.shape != bg.shape or second.shape != bg.shape:
            return None
        denom = float((first - bg).sum())
        if denom == 0:
            return np.nan
        return float((second - bg).sum() / denom)

    def reset_recapture(self) -> "ThorlabsGrabber":
        """Re-align the 3-frame grouping. Call between sequences."""
        self._recapture_buffer = []
        self._shot_phase = 0
        self._log("shot grouping reset")
        return self

    def set_saving(self, saving: bool) -> "ThorlabsGrabber":
        """Turn writing frames to disk on or off.

        Switch OFF before MOTMaster "run until stopped". Acquisition and the
        live view carry on; nothing reaches TempCameraImages, so the frames
        cannot be swept into the next run that saves.
        """
        self.saving = bool(saving)
        if self.saving:
            # Anything still sitting there predates this and would be swept
            # into the next zip.
            removed = self.clear_output_dir()
            if removed:
                self._log(
                    "saving ON, cleared {} stale file(s) first".format(removed)
                )
            else:
                self._log("saving ON")
        else:
            self._log("saving OFF - frames are displayed but not written")
        return self

    def set_compute(
        self,
        number: Optional[bool] = None,
        recapture: Optional[bool] = None
    ) -> "ThorlabsGrabber":
        """Enable or disable the per-frame derived quantities."""
        if number is not None:
            self.compute_number = bool(number)
        if recapture is not None:
            self.compute_recapture = bool(recapture)
            if self.compute_recapture:
                self._recapture_buffer = []
                self._shot_phase = 0
                # A recapture sequence is three triggers per shot.
                self.frames_per_shot = 3
        return self

    def get_latest_frame(self):
        """Return (image, index, timestamp), or (None, None, None)."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None, None, None
            return self._latest_frame, self._latest_index, self._latest_time

    def _run(self) -> None:
        consecutive_errors = 0
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                consecutive_errors = 0
            except Exception as e:
                # A single failed poll used to kill acquisition for good,
                # silently, mid-scan. Ride out a transient, give up on a
                # persistent fault, and say so loudly either way.
                self.last_error = e
                consecutive_errors += 1
                self._log("polling error {} of {}: {}".format(
                    consecutive_errors, self.max_poll_errors, e))
                if consecutive_errors >= self.max_poll_errors:
                    self._log(
                        "giving up after {} consecutive polling errors. The "
                        "camera has stopped delivering frames - check "
                        "grabber.last_error, then power-cycle the camera "
                        "(unplug the USB).".format(consecutive_errors))
                    break
                self._stop_event.wait(0.5)
                continue
            if frame is None:
                # No trigger yet. Expected between MOTMaster runs.
                continue
            try:
                image = frame.image_buffer
                if image.ndim == 1:
                    image = image.reshape(self.image_height, self.image_width)
                # image_buffer is reused by the SDK, so copy before writing.
                image = np.copy(image)
                # Crop before saving: the .tif that reaches the zip contains
                # only this region, so the analysis sees exactly what the
                # live view shows.
                crop = self.crop
                if crop is not None:
                    r0, r1, c0, c1 = crop
                    r0 = max(0, min(r0, image.shape[0] - 1))
                    r1 = max(r0 + 1, min(r1, image.shape[0]))
                    c0 = max(0, min(c0, image.shape[1] - 1))
                    c1 = max(c0 + 1, min(c1, image.shape[1]))
                    image = image[r0:r1, c0:c1]
                # Not saving still acquires and displays; it just does not
                # leave anything for MOTMaster to sweep.
                if self.saving:
                    self._write_frame(image)
                self._publish_frame(image)

                # The camera's own counter reveals frames the host failed to
                # keep up with, which would show up as missing images.
                if self._last_frame_count is not None:
                    gap = frame.frame_count - self._last_frame_count - 1
                    if gap > 0:
                        self.frames_dropped += gap
                        self._log(
                            "WARNING: camera dropped {} frame(s)".format(gap)
                        )
                self._last_frame_count = frame.frame_count
            except Exception as e:
                self.last_error = e
                self._log("write error: {}".format(e))
                break

    def start(self) -> "ThorlabsGrabber":
        if self._thread is not None and self._thread.is_alive():
            self._log("already running")
            return self
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if self.clear_stale:
            removed = self.clear_output_dir()
            if removed:
                self._log(
                    "cleared {} stale file(s) from {}".format(
                        removed, self.output_dir
                    )
                )
        self._open_camera()
        self._log_settings(note="start")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._log("running, writing to {}".format(self.output_dir))
        return self

    def stop(self) -> None:
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._camera is not None:
            try:
                self._camera.disarm()
                self._camera.dispose()
            except Exception as e:
                self._log("error closing camera: {}".format(e))
            self._camera = None
        if self._sdk is not None:
            try:
                self._sdk.dispose()
            except Exception as e:
                self._log("error closing sdk: {}".format(e))
            self._sdk = None
        self._log(
            "stopped. {} frame(s) written, {} dropped".format(
                self.frames_written, self.frames_dropped
            )
        )

    def show_monitor(self, **kwargs) -> "GrabberMonitor":
        """Open the live view window. Returns immediately; does not block."""
        if self._monitor is not None and self._monitor.is_running:
            return self._monitor
        self._monitor = GrabberMonitor(self, **kwargs)
        self._monitor.start()
        return self._monitor

    def _log_settings(self, note: str = "") -> None:
        """Record the current settings against the frame counter.

        The camera settings never reach the MOTMaster zip, so without this
        the analysis cannot tell which gain or binning a saved frame used.
        Filenames embed session and counter, so (session, from_counter) is
        enough to map any frame back to the settings it was taken with.
        """
        camera = self._camera
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session": self._session,
            "prefix": self.prefix,
            "from_counter": self._counter,
            "note": note,
            "gain_db": 0.0,
            "binx": self.binx,
            "biny": self.biny,
            "bit_depth": self.bit_depth,
            "exposure_mode": self.exposure_mode,
            "black_level": None,
            "roi": list(self.roi) if self.roi else None,
            "crop": list(self.crop) if self.crop else None,
            "sequence_frame_trigger_ms": self.sequence_frame_trigger_ms,
            "sequence_exposure_ms": self.sequence_exposure_ms,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }
        if camera is not None:
            for name, getter in (
                ("binx", lambda: camera.binx),
                ("biny", lambda: camera.biny),
                ("bit_depth", lambda: camera.bit_depth),
                ("black_level", lambda: camera.black_level),
                ("image_width", lambda: camera.image_width_pixels),
                ("image_height", lambda: camera.image_height_pixels),
            ):
                try:
                    entry[name] = getter()
                except Exception:
                    pass
            try:
                if camera.gain_range.max > 0:
                    entry["gain_db"] = float(
                        camera.convert_gain_to_decibels(camera.gain)
                    )
                    self._gain_db = entry["gain_db"]
            except Exception:
                pass
        try:
            with open(self.settings_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self._log("could not write settings log: {}".format(e))

    def reconfigure(
        self,
        gain_db: Optional[float] = None,
        binx: Optional[int] = None,
        biny: Optional[int] = None,
        black_level: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> "ThorlabsGrabber":
        """Change camera settings without restarting the grabber.

        The camera must be disarmed to change these, so polling pauses
        briefly. Any frames triggered during the pause are lost, so call this
        between runs, never mid-scan.
        """
        if self._camera is None:
            raise RuntimeError("grabber is not running; call start() first")

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        camera = self._camera
        old_binx, old_biny = self.binx, self.biny
        try:
            camera.disarm()
            if roi is not None:
                camera.roi = tuple(roi)
                self.roi = tuple(roi)
            if binx is not None:
                camera.binx = int(binx)
                self.binx = int(binx)
            if biny is not None:
                camera.biny = int(biny)
                self.biny = int(biny)
            # The crop is in BINNED pixels, so a binning change would
            # silently move the region it selects. Rescale it to keep the
            # same patch of sensor.
            if self.crop is not None and (
                    self.binx != old_binx or self.biny != old_biny):
                fy = old_biny / float(self.biny)
                fx = old_binx / float(self.binx)
                r0, r1, c0, c1 = self.crop
                self.crop = (int(round(r0 * fy)), int(round(r1 * fy)),
                             int(round(c0 * fx)), int(round(c1 * fx)))
                self._log(
                    "crop rescaled for the binning change to {}".format(
                        self.crop))
            if black_level is not None:
                camera.black_level = black_level
            if gain_db is not None:
                gain_range = camera.gain_range
                if gain_range.max <= 0:
                    raise RuntimeError(
                        "this camera has no gain control (gain_range.max=0)"
                    )
                index = camera.convert_decibels_to_gain(float(gain_db))
                index = int(min(max(index, gain_range.min), gain_range.max))
                camera.gain = index
            camera.arm(self.frames_to_buffer)
            self.image_width = camera.image_width_pixels
            self.image_height = camera.image_height_pixels
            self.bit_depth = camera.bit_depth
            self._gain_db = self._read_gain_db()
        finally:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        self._log_settings(note="reconfigure")
        achieved = ""
        try:
            if camera.gain_range.max > 0:
                achieved = ", gain {:.2f} dB".format(
                    camera.convert_gain_to_decibels(camera.gain)
                )
        except Exception:
            pass
        self._log(
            "reconfigured: {}x{} px, bin {}x{}{}".format(
                self.image_width, self.image_height,
                camera.binx, camera.biny, achieved
            )
        )
        return self

    def set_crop(
        self,
        crop: Optional[Tuple[int, int, int, int]]
    ) -> "ThorlabsGrabber":
        """Set the software crop in full-sensor coordinates (r0, r1, c0, c1).

        Takes effect on the next frame. Saved .tif files then contain only
        this region, so a background run must use the SAME crop or the
        analysis will refuse to subtract it.
        """
        self.crop = tuple(int(v) for v in crop) if crop else None
        self._log_settings(note="crop")
        self._log("crop set to {}".format(self.crop or "full frame"))
        return self

    def clear_crop(self) -> "ThorlabsGrabber":
        return self.set_crop(None)

    def set_sequence_timing(
        self,
        frame_trigger_ms: float,
        exposure_ms: float
    ) -> "ThorlabsGrabber":
        """Declare what the MOTMaster sequence is doing.

        The grabber only sees trigger edges, so it cannot work this out for
        itself. The live molecule-number plot appears only when this matches
        the calibrated condition ({} ms / {} ms).
        """.format(CALIBRATED_FRAME_TRIGGER_MS, CALIBRATED_EXPOSURE_MS)
        self.sequence_frame_trigger_ms = float(frame_trigger_ms)
        self.sequence_exposure_ms = float(exposure_ms)
        self._log_settings(note="sequence timing")
        self._log(
            "sequence declared as {:g} ms trigger / {:g} ms exposure -> "
            "molecule number {}".format(
                self.sequence_frame_trigger_ms, self.sequence_exposure_ms,
                "shown" if self.is_calibrated_condition else "hidden"
            )
        )
        return self

    def self_test(self, verbose: bool = True) -> bool:
        """Can the camera produce a frame at all, without a hardware trigger?

        Switches briefly to software triggering, fires one, and switches back.
        This separates the two causes of "waiting for first trigger":

          frame received -> camera, USB and SDK are fine, so the problem is
                            the hardware trigger path (cable, polarity, or
                            the sequence not firing)
          nothing        -> the camera itself is not delivering frames
        """
        if self._camera is None:
            raise RuntimeError("grabber is not running; call start() first")

        was_running = self.is_running
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        camera = self._camera
        got = None
        try:
            camera.disarm()
            camera.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
            camera.frames_per_trigger_zero_for_unlimited = 1
            camera.exposure_time_us = 10000
            camera.image_poll_timeout_ms = 2000
            camera.arm(2)
            camera.issue_software_trigger()
            got = camera.get_pending_frame_or_null()
        except Exception as e:
            self._log("self-test error: {}".format(e))
        finally:
            # Always put the camera back the way the grabber expects it.
            try:
                camera.disarm()
            except Exception:
                pass
            mode = (OPERATION_MODE.BULB if self.exposure_mode == "bulb"
                    else OPERATION_MODE.HARDWARE_TRIGGERED)
            camera.operation_mode = mode
            if mode == OPERATION_MODE.HARDWARE_TRIGGERED:
                camera.exposure_time_us = self.exposure_us
            camera.frames_per_trigger_zero_for_unlimited = 1
            camera.image_poll_timeout_ms = self.poll_timeout_ms
            camera.arm(self.frames_to_buffer)
            if was_running:
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

        ok = got is not None
        if verbose:
            if ok:
                print("[self_test] SOFTWARE TRIGGER WORKED - a frame "
                      "came back.")
                print("  Camera, USB link and SDK are fine, so the fault is "
                      "in the HARDWARE")
                print("  TRIGGER path: cable on the camera I/O connector, "
                      "signal level and")
                print("  polarity, or the sequence not firing.")
            else:
                print("[self_test] NO FRAME from a software trigger.")
                print("  The camera is not delivering frames at all. Restart "
                      "the grabber,")
                print("  and if that fails power-cycle it (unplug the USB).")
            print("[self_test] polling thread running again: {}".format(
                self.is_running))
        return ok

    def capabilities(self, verbose: bool = True) -> dict:
        """Report every tunable setting and its range from the camera itself.

        Works while the grabber is running (these are read-only queries). If
        it is not running, the camera is opened briefly and released again.
        Use this rather than a datasheet: it reports what this unit actually
        supports.
        """
        camera, sdk, temporary = self._camera, None, False
        if camera is None:
            sdk = TLCameraSDK()
            available = sdk.discover_available_cameras()
            if not available:
                sdk.dispose()
                raise RuntimeError(
                    "No Thorlabs camera detected (is ThorCam open?)."
                )
            camera = sdk.open_camera(self.serial_number or available[0])
            temporary = True

        fields = [
            "model", "serial_number", "firmware_version",
            "camera_sensor_type", "bit_depth",
            "sensor_width_pixels", "sensor_height_pixels",
            "sensor_pixel_width_um", "sensor_pixel_height_um",
            "sensor_pixel_size_bytes", "sensor_readout_time_ns",
            "gain", "gain_range",
            "black_level", "black_level_range",
            "binx", "binx_range", "biny", "biny_range",
            "roi", "roi_range",
            "exposure_time_us", "exposure_time_range_us",
            "frames_per_trigger_zero_for_unlimited", "frames_per_trigger_range",
            "image_width_pixels", "image_height_pixels",
            "data_rate", "is_hot_pixel_correction_enabled",
            "hot_pixel_correction_threshold",
            "hot_pixel_correction_threshold_range",
            "is_eep_supported", "is_frame_rate_control_enabled",
            "frame_rate_control_value_range",
        ]
        report = {}
        try:
            for name in fields:
                try:
                    report[name] = getattr(camera, name)
                except Exception as e:
                    report[name] = "unsupported ({})".format(
                        type(e).__name__
                    )
            for mode in OPERATION_MODE:
                if mode.name.startswith("RESERVED"):
                    continue
                try:
                    report["mode_" + mode.name] = \
                        camera.get_is_operation_mode_supported(mode)
                except Exception:
                    report["mode_" + mode.name] = "unknown"
            # Gain is reported as an index; decibels are what you reason with.
            try:
                gr = camera.gain_range
                if gr.max > 0:
                    report["gain_max_db"] = \
                        camera.convert_gain_to_decibels(gr.max)
                    report["gain_current_db"] = \
                        camera.convert_gain_to_decibels(camera.gain)
                else:
                    report["gain_max_db"] = "no gain control on this model"
            except Exception as e:
                report["gain_max_db"] = "unavailable ({})".format(
                    type(e).__name__
                )
        finally:
            if temporary:
                try:
                    camera.dispose()
                    sdk.dispose()
                except Exception:
                    pass

        if verbose:
            width = max(len(k) for k in report)
            for key, value in report.items():
                print("  {:<{w}}  {}".format(key, value, w=width))
        return report

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> str:
        pending = len(glob.glob(os.path.join(self.output_dir, "*.tif")))
        return (
            "running={} written={} dropped={} waiting_in_folder={}".format(
                self.is_running, self.frames_written,
                self.frames_dropped, pending
            )
        )

    def __enter__(self) -> "ThorlabsGrabber":
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()


class GrabberMonitor():
    """Live view window for a ThorlabsGrabber.

    Runs its own Tk main loop on a background thread so the notebook or
    console stays usable while the window is open. Every Tk call is made on
    that thread, which is what makes this safe.
    """

    def __init__(
        self,
        grabber: ThorlabsGrabber,
        refresh_ms: int = 100,
        zoom: int = 3,
        colormap: Optional[str] = "inferno",
        autoscale: bool = True,
        vmin: int = 0,
        vmax: int = 1023,
        title: str = "MOT camera monitor",
        show_number: bool = True,
        fit: bool = True,
        fit_px: int = 460,
        subtract_display: bool = False,
        stop_grabber_on_close: bool = True
    ) -> None:
        self.grabber = grabber
        self.refresh_ms = refresh_ms
        self.zoom = zoom
        self.colormap = colormap
        self.autoscale = autoscale
        self.vmin = vmin
        self.vmax = vmax
        self.title = title
        self.show_number = show_number
        # Fit mode scales the image to fill the window, so a small crop is
        # magnified rather than shown at a few pixels across.
        self.fit = fit
        self.fit_px = fit_px
        self._applied_zoom = float(zoom)
        # Display-only background subtraction. Never affects saved files.
        self.subtract_display = subtract_display
        # Closing the window shuts the camera down too. Set False to leave
        # acquisition running when the window is closed.
        self.stop_grabber_on_close = stop_grabber_on_close
        # Distinguishes the user clicking X from a programmatic stop(), so
        # grabber.stop() closing this window cannot recurse back into itself.
        self._closed_by_user = False

        self._thread = None
        self._stop_event = threading.Event()
        self._root = None
        self._photo = None
        self._shown_index = None
        self._cmap = None
        self._stats = None
        self._state = None
        self._image_canvas = None
        self._panels = {}
        self._gain_entry = None
        self._bin_entry = None
        self._notice = None
        self._autoscale_btn = None
        self._fit_btn = None
        self._subtract_btn = None
        self._number_btn = None
        self._recapture_btn = None
        self._saving_btn = None
        self._frames_entry = None
        self._bg_entry = None
        self._drag_start = None
        self._drag_rect = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "GrabberMonitor":
        if self.is_running:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _load_colormap(self):
        if self.colormap is None:
            return None
        try:
            import matplotlib
            # matplotlib.colormaps arrived in 3.5 and cm.get_cmap was removed
            # in 3.9, so try the new name first and fall back to the old one.
            try:
                return matplotlib.colormaps[self.colormap]
            except (AttributeError, KeyError):
                from matplotlib import cm
                return cm.get_cmap(self.colormap)
        except Exception:
            # Falling back to greyscale is better than losing the window.
            return None

    def _effective_zoom(self, image: np.ndarray) -> float:
        """Scale factor actually applied to the displayed image.

        In fit mode a tightly cropped region is magnified to fill the window,
        so a 25 x 25 crop is still usable rather than a postage stamp.
        """
        if not self.fit:
            return float(self.zoom)
        h, w = image.shape
        return max(1.0, min(self.fit_px / max(h, 1), self.fit_px / max(w, 1)))

    def _to_photo(self, image: np.ndarray):
        from PIL import ImageTk
        arr = image.astype(np.float64)
        if self.autoscale:
            lo, hi = float(arr.min()), float(arr.max())
        else:
            lo, hi = float(self.vmin), float(self.vmax)
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        if self._cmap is not None:
            rgb = (self._cmap(norm)[..., :3] * 255).astype(np.uint8)
            img = Image.fromarray(rgb, mode="RGB")
        else:
            img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        z = self._effective_zoom(image)
        self._applied_zoom = z
        if z != 1:
            img = img.resize(
                (max(1, int(img.size[0] * z)), max(1, int(img.size[1] * z))),
                Image.NEAREST
            )
        return ImageTk.PhotoImage(img), lo, hi

    def _build_panel(self, tk, key, title, colour, fmt, points=200):
        """One trace panel: graph, large live value, and manual scaling.

        Deliberately uses NO Tk StringVar/BooleanVar. A Tk variable that
        escapes into user code keeps the interpreter alive past teardown and
        kills the process with Tcl_AsyncDelete when it is finalised on
        another thread. Plain widgets and plain Python values avoid that.
        """
        frame = tk.Frame(self._root)
        label = tk.Label(frame, text=title, font=("Consolas", 8),
                         foreground="#555555")
        label.pack(anchor="w")

        row = tk.Frame(frame)
        row.pack(anchor="w")
        canvas = tk.Canvas(row, width=420, height=90, background="white",
                           highlightthickness=1, highlightbackground="#cccccc")
        canvas.pack(side="left")
        value = tk.Label(row, text="--", font=("Consolas", 24, "bold"),
                         foreground=colour, width=11, anchor="w")
        value.pack(side="left", padx=(12, 0))

        ctl = tk.Frame(frame)
        ctl.pack(anchor="w", pady=(2, 0))
        panel = {
            "frame": frame, "canvas": canvas, "value": value, "label": label,
            "colour": colour, "fmt": fmt, "auto": True,
        }

        def toggle_auto():
            panel["auto"] = not panel["auto"]
            panel["auto_button"].config(
                text="auto-y: on" if panel["auto"] else "auto-y: OFF")

        panel["auto_button"] = tk.Button(ctl, text="auto-y: on", width=11,
                                         command=toggle_auto)
        panel["auto_button"].pack(side="left")
        tk.Label(ctl, text=" y").pack(side="left")
        panel["ymin"] = tk.Entry(ctl, width=9)
        panel["ymin"].pack(side="left")
        tk.Label(ctl, text="to").pack(side="left")
        panel["ymax"] = tk.Entry(ctl, width=9)
        panel["ymax"].pack(side="left")
        tk.Label(ctl, text="   points").pack(side="left")
        panel["pts"] = tk.Entry(ctl, width=6)
        panel["pts"].insert(0, str(points))
        panel["pts"].pack(side="left")
        tk.Button(ctl, text="reset",
                  command=lambda k=key: self._reset_panel(k)).pack(
                      side="left", padx=(10, 0))

        self._panels[key] = panel
        return frame

    def _histories_for(self, key):
        return {
            "counts": [self.grabber.net_counts_history,
                       self.grabber.counts_history],
            "number": [self.grabber.number_history],
            "recapture": [self.grabber.recapture_history],
        }.get(key, [])

    def _reset_panel(self, key) -> None:
        """Clear this trace and return it to auto scaling."""
        for h in self._histories_for(key):
            h.clear()
        p = self._panels.get(key)
        if not p:
            return
        p["auto"] = True
        p["auto_button"].config(text="auto-y: on")
        p["ymin"].delete(0, "end")
        p["ymax"].delete(0, "end")

    def _draw_panel(self, key, values, message=None) -> None:
        p = self._panels.get(key)
        if p is None:
            return
        canvas = p["canvas"]
        canvas.delete("all")
        w, h = int(canvas["width"]), int(canvas["height"])

        if message:
            canvas.create_text(w / 2, h / 2, text=message, fill="#b35900",
                               font=("Consolas", 8), width=w - 20)
            p["value"].config(text="--")
            return

        try:
            n_show = max(2, int(float(p["pts"].get())))
        except (ValueError, TypeError):
            n_show = 200
        vals = [v for v in values if v == v][-n_show:]
        if not vals:
            canvas.create_text(w / 2, h / 2, text="waiting for data",
                               fill="#999999", font=("Consolas", 8))
            p["value"].config(text="--")
            return

        p["value"].config(text=p["fmt"].format(vals[-1]))
        if len(vals) < 2:
            canvas.create_text(w / 2, h / 2, text="waiting for data",
                               fill="#999999", font=("Consolas", 8))
            return

        lo, hi = min(vals), max(vals)
        if not p["auto"]:
            try:
                lo, hi = float(p["ymin"].get()), float(p["ymax"].get())
            except (ValueError, TypeError):
                pass          # unparseable entry: fall back to auto
        if hi <= lo:
            hi = lo + 1.0

        step = w / float(len(vals) - 1)
        coords = []
        for i, v in enumerate(vals):
            y = h - 2 - (v - lo) / (hi - lo) * (h - 6)
            # Clamp, so one wild point cannot drag the trace off the canvas.
            coords.extend([i * step, min(max(y, 1.0), h - 1.0)])
        canvas.create_line(*coords, fill=p["colour"], width=2)
        canvas.create_text(3, 7, text="{:.4g}".format(hi), anchor="w",
                           fill="#888888", font=("Consolas", 7))
        canvas.create_text(3, h - 7, text="{:.4g}".format(lo), anchor="w",
                           fill="#888888", font=("Consolas", 7))

    def _build_camera_controls(self, tk) -> None:
        """Gain and binning, applied through reconfigure().

        reconfigure() disarms the camera and restarts the polling thread, so
        it runs on a worker thread and the result is posted back with
        after(), never touching widgets from off the Tk thread.
        """
        row = tk.Frame(self._root)
        row.pack(anchor="w", padx=8, pady=(0, 8))

        tk.Label(row, text="camera:  gain").pack(side="left")
        self._gain_entry = tk.Entry(row, width=6)
        self._gain_entry.insert(0, "{:g}".format(self.grabber._gain_db))
        self._gain_entry.pack(side="left")
        tk.Label(row, text="dB   bin").pack(side="left")
        self._bin_entry = tk.Entry(row, width=4)
        self._bin_entry.insert(0, str(self.grabber.binx))
        self._bin_entry.pack(side="left")

        def done(message):
            if self._root is None:
                return
            if self._gain_entry is not None:
                self._gain_entry.delete(0, "end")
                self._gain_entry.insert(
                    0, "{:g}".format(self.grabber._gain_db))
                self._bin_entry.delete(0, "end")
                self._bin_entry.insert(0, str(self.grabber.binx))
            self._notice.config(text=message)

        def worker(gain, nbin):
            try:
                self.grabber.reconfigure(gain_db=gain, binx=nbin, biny=nbin)
                msg = ("applied: gain {:g} dB, bin {}x{} - RETAKE the "
                       "background").format(
                           self.grabber._gain_db, self.grabber.binx,
                           self.grabber.biny)
            except Exception as e:
                msg = "reconfigure failed: {}".format(e)
            root = self._root
            if root is not None:
                try:
                    root.after(0, lambda: done(msg))
                except Exception:
                    pass

        def apply_camera():
            try:
                gain = float(self._gain_entry.get())
            except (ValueError, TypeError):
                gain = None
            try:
                nbin = max(1, int(float(self._bin_entry.get())))
            except (ValueError, TypeError):
                nbin = None
            if gain is None and nbin is None:
                self._notice.config(text="gain/bin: could not read those values")
                return
            self._notice.config(text="applying camera settings...")
            threading.Thread(
                target=worker, args=(gain, nbin), daemon=True).start()

        tk.Button(row, text="apply", command=apply_camera).pack(
            side="left", padx=(8, 0))
        self._gain_entry.bind("<Return>", lambda e: apply_camera())
        self._bin_entry.bind("<Return>", lambda e: apply_camera())

    def _relayout(self) -> None:
        """Show only the panels whose toggle is on."""
        for key, on in (
            ("counts", True),
            ("number", self.grabber.compute_number),
            ("recapture", self.grabber.compute_recapture),
        ):
            p = self._panels.get(key)
            if p is None:
                continue
            if on:
                p["frame"].pack(anchor="w", padx=8, pady=(4, 6))
            else:
                p["frame"].pack_forget()

    def _draw_history(self) -> None:
        net = [v for v in self.grabber.net_counts_history if v == v]
        if len(net) >= 2:
            self._draw_panel("counts", net)
            self._panels["counts"]["label"].config(
                text="integrated counts per frame (background subtracted)")
        else:
            self._draw_panel("counts", list(self.grabber.counts_history))
            self._panels["counts"]["label"].config(
                text="integrated counts per frame (RAW - no background)")

        if self.grabber.compute_number:
            ok, reason = self.grabber.number_status()
            self._draw_panel(
                "number", list(self.grabber.number_history),
                message=None if ok else "molecule number hidden: " + reason)

        if self.grabber.compute_recapture:
            rates = list(self.grabber.recapture_history)
            msg = None
            if not [v for v in rates if v == v]:
                msg = ("waiting for a complete 3-frame shot ({} buffered)"
                       .format(len(self.grabber._recapture_buffer)))
            self._draw_panel("recapture", rates, message=msg)

    # ---- crop selection by dragging on the image ------------------------
    def _on_press(self, event) -> None:
        self._drag_start = (event.x, event.y)
        if self._drag_rect is not None:
            self._image_canvas.delete(self._drag_rect)
        self._drag_rect = None

    def _on_drag(self, event) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        if self._drag_rect is not None:
            self._image_canvas.delete(self._drag_rect)
        self._drag_rect = self._image_canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="#ffffff", width=2, dash=(4, 3)
        )

    def _on_release(self, event) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        x1, y1 = event.x, event.y
        if self._drag_rect is not None:
            self._image_canvas.delete(self._drag_rect)
            self._drag_rect = None

        # Canvas pixels -> displayed-image pixels. Must use the zoom that was
        # actually applied, which in fit mode is not self.zoom.
        z = self._applied_zoom or 1.0
        c0, c1 = sorted((int(x0 / z), int(x1 / z)))
        r0, r1 = sorted((int(y0 / z), int(y1 / z)))
        if (r1 - r0) < 4 or (c1 - c0) < 4:
            self._notice.config(text="crop ignored: drag a larger box (min 4 px)")
            return

        # The displayed image is already cropped, so offset back into
        # full-sensor coordinates before storing.
        current = self.grabber.crop
        off_r, off_c = (current[0], current[2]) if current else (0, 0)
        self.grabber.set_crop(
            (off_r + r0, off_r + r1 + 1, off_c + c0, off_c + c1 + 1)
        )
        self._shown_index = None

    def _on_window_close(self) -> None:
        """User clicked the X. Shuts the camera down too by default."""
        self._closed_by_user = True
        self._stop_event.set()

    def _clear_crop(self) -> None:
        self.grabber.clear_crop()
        self._shown_index = None

    def _apply_background_file(self) -> None:
        """Load today's run zip named in the box as the background."""
        text = self._bg_entry.get().strip()
        try:
            if text == "":
                self.grabber.set_background_file(None)
            else:
                self.grabber.set_background_file(int(text))
        except ValueError:
            # Either a non-integer, or a zip with no images. Both are
            # reported on the molecule-number panel like any other reason.
            self.grabber._background = None
            self.grabber._background_error = (
                "'{}' is not a file number".format(text)
                if not text.lstrip("-").isdigit()
                else "file {} contains no images".format(text)
            )
        except FileNotFoundError:
            pass  # set_background_file already recorded the reason
        except Exception as e:
            self.grabber._background = None
            self.grabber._background_error = str(e)[:80]
        self._shown_index = None

    def _tick(self) -> None:
        if self._stop_event.is_set():
            # Leave the mainloop and let _run tear the widgets down in this
            # same thread. Destroying here would strand the Tk variables.
            self._root.quit()
            return
        image, index, timestamp = self.grabber.get_latest_frame()
        if image is not None and index != self._shown_index:
            self._shown_index = index
            # Display-only subtraction. The saved .tif is always the raw
            # frame; nothing here can reach the file on disk.
            shown = image
            subtracted_view = False
            if self.subtract_display:
                sub = self.grabber.subtracted(image)
                if sub is not None:
                    shown, subtracted_view = sub, True
            photo, lo, hi = self._to_photo(shown)
            # Tk discards images that are not referenced, so hold on to it.
            self._photo = photo
            self._image_canvas.configure(
                width=photo.width(), height=photo.height()
            )
            self._image_canvas.delete("frame")
            self._image_canvas.create_image(
                0, 0, anchor="nw", image=photo, tags="frame"
            )
            self._image_canvas.tag_lower("frame")
            number = (
                self.grabber.number_history[-1]
                if self.grabber.number_history else float("nan")
            )
            number_text = (
                "   N {:.3e}".format(number)
                if number == number else ""
            )
            self._stats.config(text=(
                "frame {}   {}min {:.0f}   max {:.0f}   sum {:.3e}{}   "
                "scale [{:.0f}, {:.0f}]").format(
                    index, "[bg sub] " if subtracted_view else "",
                    shown.min(), shown.max(), shown.sum(),
                    number_text, lo, hi
                )
            )
            self._draw_history()
        age = "-" if timestamp is None else "{:.1f} s".format(
            time.time() - timestamp
        )
        crop = self.grabber.crop
        crop_text = (
            "full frame" if not crop
            else "crop r{}:{} c{}:{}".format(*crop)
        )
        self._state.config(text=(
            "{}   written {}   dropped {}   last frame {} ago   {}   "
            "{}   bg: {}").format(
                "RUNNING" if self.grabber.is_running else "STOPPED",
                self.grabber.frames_written, self.grabber.frames_dropped, age,
                crop_text,
                "SAVING" if self.grabber.saving else "NOT SAVING",
                self.grabber.number_status()[1]
            )
        )
        self._root.after(self.refresh_ms, self._tick)

    def _run(self) -> None:
        import tkinter as tk
        self._cmap = self._load_colormap()

        self._root = tk.Tk()
        self._root.title(self.title)
        self._root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # A Canvas rather than a Label, so the crop rectangle can be drawn
        # over the live image while dragging.
        self._image_canvas = tk.Canvas(
            self._root, background="black", highlightthickness=0,
            width=400, height=300, cursor="crosshair"
        )
        self._image_canvas.pack(padx=8, pady=8)
        self._image_canvas.bind("<Button-1>", self._on_press)
        self._image_canvas.bind("<B1-Motion>", self._on_drag)
        self._image_canvas.bind("<ButtonRelease-1>", self._on_release)

        tk.Label(
            self._root, text="drag on the image to crop; saved .tif files "
                             "contain only the cropped region",
            font=("Consolas", 8), foreground="#777777"
        ).pack(anchor="w", padx=8)

        self._stats = tk.Label(
            self._root, text="waiting for first trigger...",
            font=("Consolas", 9), anchor="w", justify="left")
        self._stats.pack(anchor="w", padx=8)

        self._state = tk.Label(
            self._root, text="", font=("Consolas", 9), foreground="#555555",
            anchor="w", justify="left")
        self._state.pack(anchor="w", padx=8)
        # Messages that must survive the next tick, unlike the status line.
        self._notice = tk.Label(
            self._root, text="", font=("Consolas", 9), foreground="#b35900",
            anchor="w", justify="left")
        self._notice.pack(anchor="w", padx=8)

        self._build_panel(
            tk, "counts", "integrated counts per frame", "#2f81f7", "{:.4g}")
        self._build_panel(
            tk, "number",
            "molecule number  (valid only at {:g} ms trigger / {:g} ms "
            "exposure; first frame of each shot only)".format(
                CALIBRATED_FRAME_TRIGGER_MS, CALIBRATED_EXPOSURE_MS),
            "#1a7f37", "{:.0f}")
        self._build_panel(
            tk, "recapture",
            "recapture rate  (3 frames per shot: first, second, background)",
            "#8250df", "{:.3f}")

        controls = tk.Frame(self._root)
        controls.pack(anchor="w", padx=8, pady=(0, 8))
        # Held on self, not in a local, so teardown below can release it
        # explicitly while the Tk interpreter is still alive.

        def toggle_autoscale():
            self.autoscale = not self.autoscale
            self._autoscale_btn.config(
                text="autoscale: on" if self.autoscale else "autoscale: OFF")
            self._shown_index = None  # force a redraw with the new scaling

        self._autoscale_btn = tk.Button(
            controls, width=14, command=toggle_autoscale,
            text="autoscale: on" if self.autoscale else "autoscale: OFF")
        self._autoscale_btn.pack(side="left")

        # Held on self, never in a local: a Tk variable captured by a closure
        # outlives teardown and crashes the interpreter on finalisation.
        def toggle_fit():
            self.fit = not self.fit
            self._fit_btn.config(text="fit: on" if self.fit else "fit: OFF")
            self._shown_index = None

        self._fit_btn = tk.Button(
            controls, width=8, command=toggle_fit,
            text="fit: on" if self.fit else "fit: OFF")
        self._fit_btn.pack(side="left", padx=(12, 0))

        def toggle_subtract():
            self.subtract_display = not self.subtract_display
            self._subtract_btn.config(
                text="subtract bg: on" if self.subtract_display
                else "subtract bg: OFF")
            self._shown_index = None

        self._subtract_btn = tk.Button(
            controls, width=18, command=toggle_subtract,
            text="subtract bg: on" if self.subtract_display
            else "subtract bg: OFF")
        self._subtract_btn.pack(side="left", padx=(8, 0))

        def zoom_by(delta):
            # Manual zoom means the user wants control, so leave fit mode.
            if self.fit:
                self.zoom = max(1, int(round(self._applied_zoom)))
                self.fit = False
                self._fit_btn.config(text="fit: OFF")
            self.zoom = max(1, min(40, self.zoom + delta))
            self._shown_index = None

        tk.Button(controls, text="-", width=2,
                  command=lambda: zoom_by(-1)).pack(side="left", padx=(6, 2))
        tk.Button(controls, text="+", width=2,
                  command=lambda: zoom_by(1)).pack(side="left")
        tk.Button(controls, text="full frame",
                  command=self._clear_crop).pack(side="left", padx=(12, 0))

        # Second row: what to compute, and whether to write to disk.
        toggles = tk.Frame(self._root)
        toggles.pack(anchor="w", padx=8, pady=(0, 6))

        def toggle_number():
            on = not self.grabber.compute_number
            self.grabber.set_compute(number=on)
            self._number_btn.config(
                text="molecule number: on" if on else "molecule number: OFF")
            self._relayout()

        def toggle_recapture():
            on = not self.grabber.compute_recapture
            self.grabber.set_compute(recapture=on)
            self._recapture_btn.config(
                text="recapture: on" if on else "recapture: OFF")
            self._frames_entry.delete(0, "end")
            self._frames_entry.insert(0, str(self.grabber.frames_per_shot))
            self._relayout()

        def toggle_saving():
            on = not self.grabber.saving
            self.grabber.set_saving(on)
            self._saving_btn.config(
                text="SAVING to disk" if on else "NOT saving")

        self._number_btn = tk.Button(
            toggles, width=21, command=toggle_number,
            text="molecule number: on" if self.grabber.compute_number
            else "molecule number: OFF")
        self._number_btn.pack(side="left")
        self._recapture_btn = tk.Button(
            toggles, width=16, command=toggle_recapture,
            text="recapture: on" if self.grabber.compute_recapture
            else "recapture: OFF")
        self._recapture_btn.pack(side="left", padx=(8, 0))
        tk.Button(toggles, text="resync",
                  command=self.grabber.reset_recapture).pack(
                      side="left", padx=(2, 10))

        # How many camera triggers per shot. Molecule number uses only the
        # first of each group, so this must match the sequence.
        tk.Label(toggles, text="frames/shot").pack(side="left")

        def apply_frames(*_):
            try:
                n = max(1, int(float(self._frames_entry.get())))
            except (ValueError, TypeError):
                return
            self.grabber.frames_per_shot = n
            self.grabber.reset_recapture()

        self._frames_entry = tk.Entry(toggles, width=4)
        self._frames_entry.insert(0, str(self.grabber.frames_per_shot))
        self._frames_entry.pack(side="left", padx=(2, 12))
        self._frames_entry.bind("<Return>", apply_frames)
        self._frames_entry.bind("<FocusOut>", apply_frames)
        self._saving_btn = tk.Button(
            toggles, width=15, command=toggle_saving,
            text="SAVING to disk" if self.grabber.saving else "NOT saving")
        self._saving_btn.pack(side="left")

        tk.Label(controls, text="  bg file:").pack(side="left")
        self._bg_entry = tk.Entry(controls, width=5)
        if self.grabber._background_file_no is not None:
            self._bg_entry.insert(0, str(self.grabber._background_file_no))
        self._bg_entry.pack(side="left")
        self._bg_entry.bind(
            "<Return>", lambda e: self._apply_background_file())
        tk.Button(controls, text="set",
                  command=self._apply_background_file).pack(
                      side="left", padx=(4, 0))

        self._build_camera_controls(tk)

        self._relayout()

        root = self._root
        self._root.after(self.refresh_ms, self._tick)
        try:
            self._root.mainloop()
        except Exception as e:
            print("[GrabberMonitor] window error: {}".format(e))
        finally:
            self._teardown(root)
            if self._closed_by_user and self.stop_grabber_on_close:
                # Clear the back-reference first: grabber.stop() stops its
                # monitor, and joining this thread from inside itself would
                # raise "cannot join current thread".
                try:
                    self.grabber._monitor = None
                    self.grabber.stop()
                except Exception as e:
                    print("[GrabberMonitor] error stopping grabber: "
                          "{}".format(e))

    def _teardown(self, root) -> None:
        """Release every Tk object from the thread that created them.

        Any Tk object finalised after the interpreter is gone, or on another
        thread, kills the process with "Tcl_AsyncDelete: async handler
        deleted by the wrong thread" - in a notebook, taking the kernel with
        it. The GUI therefore uses no Tk StringVar/BooleanVar at all (plain
        widgets and plain Python values instead), and everything it does hold
        is released here, on the Tk thread, before the interpreter dies.
        """
        import gc
        self._photo = None
        self._stats = None
        self._state = None
        self._image_canvas = None
        self._panels = {}
        self._gain_entry = None
        self._bin_entry = None
        self._notice = None
        self._autoscale_btn = None
        self._fit_btn = None
        self._subtract_btn = None
        self._number_btn = None
        self._recapture_btn = None
        self._saving_btn = None
        self._frames_entry = None
        self._bg_entry = None
        gc.collect()
        try:
            root.destroy()
        except Exception:
            pass
        self._root = None
        gc.collect()


def set_run_until_stopped(experiment, grabber, state: bool):
    """Put MOTMaster into run-until-stopped and stop saving frames, together.

    MOTMaster never saves a zip in that mode, so it never sweeps
    TempCameraImages and the frames would otherwise pile up and land in the
    next run that does save. There is no getter for the mode over remoting,
    so the two have to be set as a pair.

        set_run_until_stopped(expt, grabber, True)   # align, watch live
        set_run_until_stopped(expt, grabber, False)  # back to saving runs
    """
    experiment.motmaster.SetRunUntilStopped(bool(state))
    grabber.set_saving(not state)
    return grabber


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Free-running Thorlabs grabber for MOTMaster."
    )
    parser.add_argument("--serial", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--exposure-mode", default="bulb", choices=["bulb", "fixed"],
        help="bulb: trigger level sets exposure; fixed: use --exposure-us"
    )
    parser.add_argument("--exposure-us", type=int, default=10000)
    parser.add_argument("--bin", type=int, default=8, dest="bin_")
    parser.add_argument("--binx", type=int, default=None)
    parser.add_argument("--biny", type=int, default=None)
    parser.add_argument(
        "--roi", type=int, nargs=4, default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="ROI in unbinned sensor pixels: upper-left x y, lower-right x y"
    )
    parser.add_argument("--trigger-polarity", default="active_high",
                        choices=["active_high", "active_low"])
    parser.add_argument("--monitor", action="store_true",
                        help="open the live view window")
    args = parser.parse_args()

    grabber = ThorlabsGrabber(
        serial_number=args.serial,
        output_dir=args.output_dir,
        prefix=args.prefix,
        exposure_mode=args.exposure_mode,
        exposure_us=args.exposure_us,
        binx=args.binx if args.binx is not None else args.bin_,
        biny=args.biny if args.biny is not None else args.bin_,
        roi=tuple(args.roi) if args.roi else None,
        trigger_polarity=args.trigger_polarity,
    )
    try:
        grabber.start()
    except Exception as e:
        print("Failed to start: {}".format(e))
        return 1
    if args.monitor:
        grabber.show_monitor()
    print("Grabber running. Press Ctrl-C to stop.")
    try:
        while grabber.is_running:
            time.sleep(2.0)
            print(grabber.status())
    except KeyboardInterrupt:
        print()
    finally:
        grabber.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
