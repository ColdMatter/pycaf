"""
Thorlabs-camera-specific analysis: molecule number and signal-to-noise.

This module is additive and self-contained. It does NOT modify or replace
analysis3.py or pycaf.analysis.utils, so switching back to the Hamamatsu
changes nothing: just stop using these functions.

It deliberately does not import the Thorlabs SDK, so it can be imported on
any machine for offline analysis.


Why a separate molecule-number function
---------------------------------------
calculate_molecule_number_from_fluorescent_images() in pycaf.analysis.utils
does:

    photon = count * FWC / ((2**bits - 1) * eta_q)

which converts counts (ADU) to electrons via FWC/(2**bits - 1), then to
photons via the quantum efficiency. That conversion assumes full-scale ADU
corresponds to a full well, which is only true at unity gain. The Hamamatsu
ran at Gain1 = 1.00 (confirmed from the TIFF metadata in the old images), so
for that camera the assumption held and the function was correct.

The Thorlabs camera has a gain control, so the same count means fewer
electrons when the gain is raised. molecule_number() below is the same
physics with an explicit gain term, and reduces to exactly the old formula at
0 dB. Exposure is held at 10 ms to match the Hamamatsu data.
"""

from typing import Optional, Tuple, Dict, Any, List
import os
import re
import json
import numpy as np

DEFAULT_SETTINGS_LOG = r"C:\Users\cafmot\Documents\thorlabs_camera_log.jsonl"


# --------------------------------------------------------------------------
# Thorlabs CS165MU1/M (Zelux) sensor constants.
#
# Source: CS165_Series_Camera_User_Guide.pdf, sections 5.1 to 5.3, installed at
#   C:\Program Files\Thorlabs\Scientific Imaging\Documentation
#       \Scientific Camera Documents\CS165_Series_Camera_User_Guide.pdf
#
#   Sensor Type        CMOS Monochrome, global shutter
#   Active pixels      1440 (H) x 1080 (V)
#   Pixel size         3.45 um x 3.45 um
#   Optical format     4968 um x 3726 um
#   Full Well Capacity 11,000 e-
#   ADC Resolution     10 bits
#   Read Noise         < 4.0 e- RMS
#   Dynamic Range      up to 69 dB
#   Binning            digital, 1x1 to 16x16
#
# For comparison the Hamamatsu ORCA-R2 used 18,000 e- and 12 bits.
# --------------------------------------------------------------------------
CS165MU1_FULL_WELL_CAPACITY = 11000.0   # electrons, from the user guide
CS165MU1_BIT_DEPTH = 10                 # ADC resolution, from the user guide
CS165MU1_READ_NOISE_E = 4.0             # e- RMS, upper bound from the guide
CS165MU1_PIXEL_SIZE = 3.45e-6           # metres

# The user guide gives quantum efficiency ONLY as a plot (section 5.2,
# "CS165MU Series Quantum Efficiency", page 22) with no tabulated values, so
# this cannot be read out programmatically. The value below is a deliberately
# conservative estimate for the CaF imaging wavelength (~606 nm) and is the
# one number here that is NOT from the datasheet text.
#
# Being conservative here means molecule number is biased HIGH: N scales as
# 1/eta_q, so if the true QE is larger than this, the real N is lower than
# reported. Erring this way is the safe direction for a number you do not
# want to overstate downstream.
#
# It scales absolute molecule number linearly, but it is CONSTANT across gain,
# binning and ROI, so it cancels completely when comparing camera settings.
# The SNR optimisation is therefore valid before this is pinned down; only
# absolute N depends on it.
CS165MU1_ETA_Q_606NM = 0.66             # conservative estimate - see above

DEFAULT_FULL_WELL_CAPACITY = CS165MU1_FULL_WELL_CAPACITY
DEFAULT_ETA_Q = CS165MU1_ETA_Q_606NM
DEFAULT_BIT_DEPTH = CS165MU1_BIT_DEPTH
DEFAULT_EXPOSURE_TIME = 10e-3          # fixed at 10 ms to match HamCam
DEFAULT_GAMMA = 1.5e6                  # photon scattering rate, s^-1
DEFAULT_SOLID_ANGLE = 0.023            # collected fraction of 4 pi


def calculate_molecule_number_from_fluorescent_images_ThorCam(
    images: np.ndarray,
    full_well_capacity: float = CS165MU1_FULL_WELL_CAPACITY,
    bits_per_channel: int = CS165MU1_BIT_DEPTH,
    exposure_time: float = DEFAULT_EXPOSURE_TIME,
    gamma: float = DEFAULT_GAMMA,
    collection_solid_angle: float = DEFAULT_SOLID_ANGLE,
    eta_q: float = CS165MU1_ETA_Q_606NM,
    gain_db: float = 0.0
) -> np.ndarray:
    """Thorlabs CS165MU1 version of the pycaf fluorescence molecule number.

    Same argument order as
    pycaf.analysis.utils.calculate_molecule_number_from_fluorescent_images,
    so it is a drop-in replacement, with two differences:

      * the defaults are the Thorlabs sensor values (11,000 e-, 10 bit)
        rather than the Hamamatsu ones (18,000 e-, 12 bit);
      * gain_db is applied, because this camera has a gain control whereas
        the Hamamatsu ran at Gain1 = 1.00 (confirmed from the TIFF metadata
        embedded in the old images).

    At gain_db = 0 the arithmetic is identical to the original function.

    images must already be background-subtracted.

    Note that count is summed over the WHOLE frame, so any part of the cloud
    falling outside the sensor or ROI is simply lost. The Thorlabs sensor is
    4968 x 3726 um against the ORCA-R2's 8669 x 6605 um, so with unchanged
    optics the field of view is about 1.75x smaller in each direction.
    """
    images = np.asarray(images, dtype=float)
    if images.ndim == 2:
        images = images[None, ...]
    gain_linear = 10.0 ** (float(gain_db) / 20.0)
    count = np.sum(images, axis=(1, 2))
    photon = (
        (count * full_well_capacity)
        / ((2 ** bits_per_channel - 1) * eta_q * gain_linear)
    )
    number = photon / (exposure_time * gamma * collection_solid_angle)
    return number


class ThorlabsSettings():
    """The camera settings that affect how counts map to molecule number."""

    def __init__(
        self,
        gain_db: float = 0.0,
        binx: int = 8,
        biny: int = 8,
        bit_depth: int = DEFAULT_BIT_DEPTH,
        exposure_time: float = DEFAULT_EXPOSURE_TIME,
        full_well_capacity: float = DEFAULT_FULL_WELL_CAPACITY,
        eta_q: float = DEFAULT_ETA_Q,
        black_level: float = 0.0
    ) -> None:
        self.gain_db = float(gain_db)
        self.binx = int(binx)
        self.biny = int(biny)
        self.bit_depth = int(bit_depth)
        self.exposure_time = float(exposure_time)
        self.full_well_capacity = float(full_well_capacity)
        self.eta_q = float(eta_q)
        self.black_level = float(black_level)

    @property
    def gain_linear(self) -> float:
        """Gain as an amplitude factor. 0 dB -> 1.0, i.e. the HamCam case."""
        return 10.0 ** (self.gain_db / 20.0)

    @property
    def electrons_per_adu(self) -> float:
        """How many electrons one count represents at these settings."""
        return (
            self.full_well_capacity
            / ((2 ** self.bit_depth - 1) * self.gain_linear)
        )

    def __repr__(self) -> str:
        return (
            "ThorlabsSettings(gain_db={:g}, bin={}x{}, bit_depth={}, "
            "exposure={:g} s)".format(
                self.gain_db, self.binx, self.biny,
                self.bit_depth, self.exposure_time
            )
        )

    def key(self) -> Tuple:
        """Hashable identity, for grouping shots by setting."""
        return (self.gain_db, self.binx, self.biny, self.bit_depth)


def counts_to_number(
    count_adu: float,
    settings: ThorlabsSettings,
    gamma: float = DEFAULT_GAMMA,
    collection_solid_angle: float = DEFAULT_SOLID_ANGLE
) -> float:
    """Convert a summed count in ADU to a molecule number."""
    electrons = count_adu * settings.electrons_per_adu
    photons = electrons / settings.eta_q
    return photons / (
        settings.exposure_time * gamma * collection_solid_angle
    )


def molecule_number(
    images: np.ndarray,
    settings: ThorlabsSettings,
    gamma: float = DEFAULT_GAMMA,
    collection_solid_angle: float = DEFAULT_SOLID_ANGLE
) -> np.ndarray:
    """Molecule number per image, gain-aware.

    images must already be background-subtracted. At gain_db=0 this is
    numerically identical to the existing
    calculate_molecule_number_from_fluorescent_images().
    """
    images = np.atleast_3d(np.asarray(images, dtype=float))
    if images.ndim == 2:
        images = images[None, ...]
    count = np.sum(images, axis=(1, 2))
    electrons = count * settings.electrons_per_adu
    photons = electrons / settings.eta_q
    return photons / (
        settings.exposure_time * gamma * collection_solid_angle
    )


def find_mot_region(
    image: np.ndarray,
    half_size: int = 20,
    smooth: int = 2
) -> Tuple[int, int, int, int]:
    """Locate the cloud and return (row0, row1, col0, col1).

    Uses the brightest smoothed pixel, which is robust enough for a MOT on a
    flat background and avoids hard-coded crops that break when the camera or
    optics change.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim == 3:
        img = img.mean(axis=0)
    work = img - np.median(img)
    if smooth > 0:
        # Cheap box smooth; avoids a scipy dependency.
        k = 2 * smooth + 1
        pad = np.pad(work, smooth, mode="edge")
        acc = np.zeros_like(work)
        for dy in range(k):
            for dx in range(k):
                acc += pad[dy:dy + work.shape[0], dx:dx + work.shape[1]]
        work = acc / (k * k)
    iy, ix = np.unravel_index(np.argmax(work), work.shape)
    r0 = max(0, iy - half_size)
    r1 = min(img.shape[0], iy + half_size + 1)
    c0 = max(0, ix - half_size)
    c1 = min(img.shape[1], ix + half_size + 1)
    return int(r0), int(r1), int(c0), int(c1)


def _region_mask(shape: Tuple[int, int], region: Tuple[int, int, int, int]):
    mask = np.zeros(shape, dtype=bool)
    r0, r1, c0, c1 = region
    mask[r0:r1, c0:c1] = True
    return mask


def analyse_shot(
    image: np.ndarray,
    settings: ThorlabsSettings,
    signal_region: Tuple[int, int, int, int],
    gamma: float = DEFAULT_GAMMA,
    collection_solid_angle: float = DEFAULT_SOLID_ANGLE
) -> Dict[str, Any]:
    """Signal, noise and molecule number for one background-subtracted frame.

    The noise is built from two independent contributions, both in electrons:

      shot noise      sqrt(S_e)          -- unavoidable, scales with signal
      background      sigma_pix * sqrt(n_pix) -- read noise, dark, and the
                                            shot noise of any stray light

    sigma_pix is measured from the pixels of this same frame OUTSIDE the
    signal region, so it is a genuine per-shot measurement at these exact
    camera settings and never sees the MOT itself. That is what makes the
    result insensitive to molecule number fluctuating between shots.
    """
    img = np.asarray(image, dtype=float)
    e_per_adu = settings.electrons_per_adu

    sig_mask = _region_mask(img.shape, signal_region)
    n_pix = int(sig_mask.sum())

    signal_adu = float(img[sig_mask].sum())
    signal_e = signal_adu * e_per_adu

    # Noise reference: everything outside the signal box.
    outside = img[~sig_mask]
    sigma_pix_adu = float(outside.std())
    sigma_pix_e = sigma_pix_adu * e_per_adu

    shot_noise_e = np.sqrt(max(signal_e, 0.0))
    background_noise_e = sigma_pix_e * np.sqrt(n_pix)
    total_noise_e = float(np.hypot(shot_noise_e, background_noise_e))

    snr = signal_e / total_noise_e if total_noise_e > 0 else np.nan

    # Molecule number from the SAME pixels the signal and noise came from, so
    # that SNR is literally N / delta_N: the precision of this measurement.
    # The whole-frame number is also reported, because that is what the
    # existing pipeline computes and it captures any cloud wings outside the
    # box, at the cost of the background noise of every extra pixel.
    n_molecules = counts_to_number(
        signal_adu, settings, gamma, collection_solid_angle
    )
    n_molecules_frame = counts_to_number(
        float(img.sum()), settings, gamma, collection_solid_angle
    )
    delta_n = (
        n_molecules / snr if (snr and np.isfinite(snr) and snr != 0)
        else np.nan
    )

    return {
        "n_molecules_frame": float(n_molecules_frame),
        "delta_n": float(delta_n),
        "signal_adu": signal_adu,
        "signal_e": signal_e,
        "sigma_pix_adu": sigma_pix_adu,
        "shot_noise_e": float(shot_noise_e),
        "background_noise_e": float(background_noise_e),
        "total_noise_e": total_noise_e,
        "snr": float(snr),
        "n_molecules": float(n_molecules),
        # Normalised figures of merit. Which one is flat against N tells you
        # the regime: snr_per_molecule is flat when background/read limited,
        # snr_per_sqrt_molecule is flat when shot limited.
        "snr_per_molecule": (
            float(snr / n_molecules) if n_molecules > 0 else np.nan
        ),
        "snr_per_sqrt_molecule": (
            float(snr / np.sqrt(n_molecules)) if n_molecules > 0 else np.nan
        ),
        "shot_noise_fraction": (
            float(shot_noise_e ** 2 / total_noise_e ** 2)
            if total_noise_e > 0 else np.nan
        ),
        "n_pix": n_pix,
    }


def analyse_run(
    images: np.ndarray,
    background: np.ndarray,
    settings: ThorlabsSettings,
    signal_region: Optional[Tuple[int, int, int, int]] = None,
    half_size: int = 20,
    gamma: float = DEFAULT_GAMMA,
    collection_solid_angle: float = DEFAULT_SOLID_ANGLE
) -> Dict[str, Any]:
    """Analyse every frame of a run against a background run.

    images     : (n, h, w) from CaF.readFromZip(fileno)
    background : (m, h, w) from CaF.readFromZip(bg_fileno); its mean is used
    """
    images = np.asarray(images, dtype=float)
    background = np.asarray(background, dtype=float)
    if images.ndim == 2:
        images = images[None, ...]
    if background.ndim == 2:
        background = background[None, ...]
    if background.shape[1:] != images.shape[1:]:
        raise ValueError(
            "background frames are {} but images are {} - the background run "
            "must use the same binning/ROI as the data run".format(
                background.shape[1:], images.shape[1:]
            )
        )

    subtracted = images - background.mean(axis=0)
    if signal_region is None:
        signal_region = find_mot_region(
            subtracted.mean(axis=0), half_size=half_size
        )

    shots = [
        analyse_shot(frame, settings, signal_region, gamma,
                     collection_solid_angle)
        for frame in subtracted
    ]

    def col(name):
        return np.array([s[name] for s in shots], dtype=float)

    n = col("n_molecules")
    snr = col("snr")
    # A shot whose net signal came out at or below zero gives a meaningless
    # ratio. Drop those from the summary rather than letting one bad shot
    # turn the whole run's figure of merit into nan.
    n_valid = int(np.sum(np.isfinite(col("snr_per_molecule"))))
    return {
        "settings": settings,
        "signal_region": signal_region,
        "images": subtracted,
        "shots": shots,
        "n_molecules": n,
        "snr": snr,
        "snr_per_molecule": col("snr_per_molecule"),
        "snr_per_sqrt_molecule": col("snr_per_sqrt_molecule"),
        "shot_noise_fraction": col("shot_noise_fraction"),
        "n_molecules_frame": col("n_molecules_frame"),
        "delta_n": col("delta_n"),
        # delta_N is the uncertainty in molecule number from this measurement.
        # It is the figure of merit to MINIMISE when choosing camera settings:
        # unlike SNR it does not grow just because the MOT happened to be
        # bigger on that shot.
        "delta_n_mean": float(np.nanmean(col("delta_n"))),
        # Summary
        "n_shots": int(len(n)),
        "n_valid_shots": n_valid,
        "n_mean": float(np.nanmean(n)),
        "n_std": float(np.nanstd(n)),
        "snr_mean": float(np.nanmean(snr)),
        "snr_std": float(np.nanstd(snr)),
        "snr_per_molecule_mean": float(np.nanmean(col("snr_per_molecule"))),
        "snr_per_sqrt_molecule_mean": float(
            np.nanmean(col("snr_per_sqrt_molecule"))
        ),
    }


_FRAME_RE = re.compile(r"^(?P<prefix>[A-Za-z]+)(?P<session>\d+)_(?P<counter>\d+)\.tif$")


def load_settings_log(path: str = DEFAULT_SETTINGS_LOG) -> List[dict]:
    """Read the grabber's settings log, newest entries last."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "No camera settings log at {}. It is written by "
            "ThorlabsGrabber.start() and .reconfigure().".format(path)
        )
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue
    return entries


def settings_for_frames(
    frame_names: List[str],
    log: Optional[List[dict]] = None,
    log_path: str = DEFAULT_SETTINGS_LOG,
    **overrides
) -> ThorlabsSettings:
    """Work out which camera settings a zip's frames were taken with.

    frame_names are the .tif names from the zip, e.g. CImage144534_000010.tif.
    Each name carries the grabber session and a frame counter, and the log
    records the settings in force from each counter onward.

    Raises if the frames in one zip span a settings change, because averaging
    across a gain change would be silently wrong.
    """
    log = load_settings_log(log_path) if log is None else log

    parsed = []
    for name in frame_names:
        m = _FRAME_RE.match(os.path.basename(name))
        if m:
            parsed.append((m.group("session"), int(m.group("counter"))))
    if not parsed:
        raise ValueError(
            "No frame names matched the grabber naming scheme "
            "(prefix + session + _ + counter). Were these Thorlabs frames?"
        )

    chosen = []
    for session, counter in parsed:
        candidates = [
            e for e in log
            if str(e.get("session")) == session
            and int(e.get("from_counter", 0)) <= counter
        ]
        if not candidates:
            raise ValueError(
                "No settings logged for session {} counter {}. The log only "
                "covers grabber runs started after logging was added.".format(
                    session, counter
                )
            )
        # Ties on from_counter mean several settings changes landed
        # before any frame was taken; the LAST one is the one in force,
        # and max() would return the first. Index breaks the tie.
        chosen.append(max(
            enumerate(candidates),
            key=lambda t: (int(t[1]["from_counter"]), t[0]))[1])

    keys = {
        (e.get("gain_db"), e.get("binx"), e.get("biny"), e.get("bit_depth"))
        for e in chosen
    }
    if len(keys) > 1:
        raise ValueError(
            "These frames span a camera settings change ({} distinct "
            "settings). Re-run without reconfiguring mid-scan.".format(
                len(keys)
            )
        )

    entry = chosen[0]
    kwargs = dict(
        gain_db=entry.get("gain_db", 0.0) or 0.0,
        binx=entry.get("binx", 8) or 8,
        biny=entry.get("biny", 8) or 8,
        bit_depth=entry.get("bit_depth") or DEFAULT_BIT_DEPTH,
        black_level=entry.get("black_level") or 0.0,
    )
    kwargs.update(overrides)
    return ThorlabsSettings(**kwargs)


def settings_for_zip(
    zip_path: str,
    log_path: str = DEFAULT_SETTINGS_LOG,
    **overrides
) -> ThorlabsSettings:
    """Camera settings for a MOTMaster run zip."""
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".tif")]
    return settings_for_frames(names, log_path=log_path, **overrides)


def compare_settings(results: List[Dict[str, Any]]) -> str:
    """Rank several analyse_run() results by SNR per molecule."""
    rows = sorted(
        results, key=lambda r: r["snr_per_molecule_mean"], reverse=True
    )
    lines = [
        "{:>8} {:>7} {:>11} {:>9} {:>12} {:>12}".format(
            "gain/dB", "bin", "N", "SNR", "SNR/N", "shot-lim"
        )
    ]
    for r in rows:
        s = r["settings"]
        lines.append(
            "{:>8.1f} {:>7} {:>11.3e} {:>9.1f} {:>12.3e} {:>11.0%}".format(
                s.gain_db, "{}x{}".format(s.binx, s.biny),
                r["n_mean"], r["snr_mean"], r["snr_per_molecule_mean"],
                float(np.nanmean(r["shot_noise_fraction"]))
            )
        )
    return "\n".join(lines)


def get_fl_num_thorcam(
    read_from_zip,
    file_start: int,
    file_end: int,
    bg_fileno: int,
    parameter_name: str,
    settings: Optional[ThorlabsSettings] = None,
    filepath_fn=None,
    log_path: str = DEFAULT_SETTINGS_LOG,
    frames_per_shot: int = 1,
    frame_index: int = 0,
    crop: Optional[Tuple[int, int, int, int]] = None,
    gamma: float = DEFAULT_GAMMA,
    collection_solid_angle: float = DEFAULT_SOLID_ANGLE,
    show_images: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ThorCam version of the notebook's get_fl_num.

    Differences from the original, all of which matter on this camera:

      * molecule number uses the Thorlabs sensor constants and the GAIN the
        frames were actually taken at. With the Hamamatsu constants and no
        gain term, N is wrong by 2.41x at 0 dB and 1.65x the other way at
        12 dB;
      * frame selection is explicit. The original used images[1::2], which
        assumes two camera triggers per shot. The current script fires one,
        so that expression silently discards half the frames;
      * the crop is (row0, row1, col0, col1) and is applied correctly. The
        original unpacked crop_yzip1 but used crop_y1, so to_crop=True
        raised NameError.

    read_from_zip : callable taking a file number, returning (images, params),
                    i.e. CaF.readFromZip
    filepath_fn   : optional callable taking a file number and returning the
                    zip path, i.e. CaF.getFilepath. When given and settings is
                    None, the camera settings for each file are read from the
                    grabber's settings log, so gain and binning are picked up
                    automatically.
    frames_per_shot / frame_index : which frame of each shot to use. With one
                    camera trigger the defaults take every frame.
    """
    if settings is None and filepath_fn is None:
        raise ValueError(
            "Pass settings=ThorlabsSettings(gain_db=..., binx=..., ...) or "
            "filepath_fn=CaF.getFilepath so the gain can be read from the "
            "settings log. Defaulting the gain would silently misreport N."
        )

    def _settings_for(fileno):
        if settings is not None:
            return settings
        return settings_for_zip(filepath_fn(fileno), log_path=log_path)

    def _select(images):
        picked = images[frame_index::frames_per_shot]
        if len(picked) == 0:
            raise ValueError(
                "No frames selected: {} frame(s) in the file with "
                "frames_per_shot={} frame_index={}".format(
                    len(images), frames_per_shot, frame_index
                )
            )
        return picked

    bg_images, _ = read_from_zip(bg_fileno)
    bg = _select(np.asarray(bg_images, dtype=float)).mean(axis=0)

    t, n_container, n_err_container = [], [], []
    for fileno in range(file_start, file_end + 1):
        images, params = read_from_zip(fileno)
        im = _select(np.asarray(images, dtype=float)) - bg
        if crop is not None:
            r0, r1, c0, c1 = crop
            im = im[:, r0:r1, c0:c1]

        n = molecule_number(
            im, _settings_for(fileno), gamma, collection_solid_angle
        )
        n_container.append(float(np.mean(n)))
        n_err_container.append(float(np.std(n) / np.sqrt(len(n))))
        t.append(params[parameter_name])

        if show_images:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            m = ax.imshow(im.mean(axis=0))
            fig.colorbar(m, ax=ax)
            ax.set_title("{} = {}   N = {:.3e}".format(
                parameter_name, params[parameter_name], n_container[-1]))
            plt.show()

    return np.array(t), np.array(n_container), np.array(n_err_container)


def optimise_region_size(
    images: np.ndarray,
    background: np.ndarray,
    settings: ThorlabsSettings,
    half_sizes: Optional[List[int]] = None,
    gamma: float = DEFAULT_GAMMA,
    collection_solid_angle: float = DEFAULT_SOLID_ANGLE,
    verbose: bool = True
) -> Dict[str, Any]:
    """Find the summing box that measures molecule number most precisely.

    This is the cheapest SNR knob there is: no hardware change at all.

    Noise on a summed signal grows as sqrt(number of pixels), while signal
    only comes from where the cloud actually is. So every background pixel
    inside the box adds noise and no signal. Shrinking the box improves
    precision until it starts clipping the cloud, and there is an optimum in
    between. When the measurement is background limited, as this one is, the
    improvement can be several-fold.

    The box is chosen by MAXIMISING SNR on the summed signal, the standard
    optimal-aperture criterion. Note that delta_N is NOT the right criterion
    here even though it is the right one for comparing camera settings at a
    fixed box: shrinking the box lowers delta_N simply because it measures a
    smaller N, so minimising it always collapses onto the smallest box and
    measures the core of the cloud rather than the whole thing.

    capture is each box's share of the whole-frame signal, so you can see how
    much of the cloud a box actually contains.

    Returns the best half_size by maximum SNR, plus the whole curve.
    """
    if half_sizes is None:
        half_sizes = [4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 67]

    images = np.asarray(images, dtype=float)
    background = np.asarray(background, dtype=float)
    subtracted = images - background.mean(axis=0)
    centre = find_mot_region(subtracted.mean(axis=0), half_size=1)
    row = (centre[0] + centre[1]) // 2
    col = (centre[2] + centre[3]) // 2

    frame_signal = float(np.nanmean(subtracted.sum(axis=(1, 2))))

    rows = []
    for hs in half_sizes:
        region = (
            max(0, row - hs), min(subtracted.shape[1], row + hs + 1),
            max(0, col - hs), min(subtracted.shape[2], col + hs + 1),
        )
        shots = [
            analyse_shot(f, settings, region, gamma, collection_solid_angle)
            for f in subtracted
        ]
        signal = float(np.nanmean([s["signal_adu"] for s in shots]))
        rows.append({
            "half_size": hs,
            "n_pix": shots[0]["n_pix"],
            "n_molecules": float(np.nanmean([s["n_molecules"] for s in shots])),
            "snr": float(np.nanmean([s["snr"] for s in shots])),
            "delta_n": float(np.nanmean([s["delta_n"] for s in shots])),
            "capture": signal / frame_signal if frame_signal else np.nan,
        })

    valid = [r for r in rows if np.isfinite(r["snr"])]
    best = max(valid, key=lambda r: r["snr"]) if valid else None

    if verbose:
        print("{:>10} {:>8} {:>9} {:>12} {:>9} {:>12}".format(
            "half_size", "n_pix", "capture", "N", "SNR", "delta_N"))
        for r in rows:
            mark = "  <-- best SNR" if best and r is best else ""
            print("{:>10} {:>8} {:>8.0%} {:>12.3e} {:>9.1f} {:>12.3e}{}".format(
                r["half_size"], r["n_pix"], r["capture"], r["n_molecules"],
                r["snr"], r["delta_n"], mark))
        if best:
            print("\nbest half_size = {} ({} x {} px box), SNR = {:.1f}, "
                  "capturing {:.0%} of the frame signal".format(
                      best["half_size"], 2 * best["half_size"] + 1,
                      2 * best["half_size"] + 1, best["snr"],
                      best["capture"]))
            if best["half_size"] == max(half_sizes):
                print(
                    "\nWARNING: SNR is still rising at the largest box, so "
                    "this is not a real optimum. That means signal is still "
                    "accumulating out to the frame edge, which usually means "
                    "the background subtraction is leaving a broad positive "
                    "residual rather than the cloud genuinely being that "
                    "wide. Check that the background run was taken under the "
                    "same conditions as the data."
                )

    return {"rows": rows, "best": best, "centre": (row, col),
            "frame_signal": frame_signal}


def summarise(results: Dict[str, Any]) -> str:
    """One-line human summary of an analyse_run() result."""
    s = results["settings"]
    dropped = results["n_shots"] - results["n_valid_shots"]
    note = "" if dropped == 0 else "   ({} of {} shots dropped)".format(
        dropped, results["n_shots"]
    )
    return (
        "gain {:>5.1f} dB  bin {}x{}   N = {:.3e} +/- {:.1e}   "
        "SNR = {:6.1f} +/- {:4.1f}   SNR/N = {:.3e}   "
        "shot-limited {:.0%}{}".format(
            s.gain_db, s.binx, s.biny,
            results["n_mean"], results["n_std"],
            results["snr_mean"], results["snr_std"],
            results["snr_per_molecule_mean"],
            float(np.nanmean(results["shot_noise_fraction"])),
            note,
        )
    )
