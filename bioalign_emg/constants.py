from __future__ import annotations

import re

FS = 200.0
WINDOW_LEN = 50
WINDOW_STEP = 10
N_CHANNELS = 8
N_CLASSES = 7

GESTURES = [
    "eversion",
    "fist",
    "open_hand",
    "pinch_forefinger",
    "pinch_middlefinger",
    "two",
    "varus",
]
GESTURE_TO_INDEX = {gesture: index for index, gesture in enumerate(GESTURES)}

FILENAME_RE = re.compile(
    r"^emg_p(?P<position>\d+)_r(?P<rep>\d+)_(?P<gesture>.+)\.csv$",
    re.IGNORECASE,
)

PRIMARY_MODELS = [
    "TCN_plain",
    "TCN_RingAug",
    "SE_RingAug",
    "CBAM_RingAug",
    "BioAlign_RingAug",
]

DEFAULT_SEEDS = [42, 2026, 3407]
EXPECTED_SESSION0_TRIALS = 11 * 7 * 3
