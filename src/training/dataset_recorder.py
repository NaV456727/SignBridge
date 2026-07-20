import time
import pandas as pd

from src.config import (
    DATASET_FILE,
    GESTURES,
    TARGET_SAMPLES,
    SAVE_INTERVAL,
)

from src.training.csv_writer import CSVWriter


class DatasetRecorder:

    def __init__(self):

        self.writer = CSVWriter(DATASET_FILE)

        self.current_label = None

        self.sample_count = 0

        self.recording = False

        self.countdown = False

        self.countdown_start = 0

        self.last_save = 0

        self.previous_landmarks = None

    # -------------------------------------------------

    def get_existing_samples(self, label):

        try:

            data = pd.read_csv(DATASET_FILE)

            return len(
                data[data["label"] == label]
            )

        except Exception:

            return 0

    # -------------------------------------------------

    def process_key(self, key):

        if key == 255:
            return

        key = chr(key)

        if key not in GESTURES:
            return

        selected_label = GESTURES[key]

        # Don't restart current recording
        if (
            selected_label == self.current_label
            and (self.recording or self.countdown)
        ):
            return

        self.current_label = selected_label

        self.sample_count = self.get_existing_samples(
            self.current_label
        )

        self.recording = False

        self.countdown = True

        self.countdown_start = time.time()

        self.previous_landmarks = None

        print(f"\nSelected Gesture : {self.current_label}")

    # -------------------------------------------------

    def update(self, hands):

        if self.current_label is None:
            return

        # ---------------- Countdown ----------------

        if self.countdown:

            if time.time() - self.countdown_start >= 3:

                self.countdown = False

                self.recording = True

                self.last_save = time.time()

                print("Recording Started!")

            return

        if not self.recording:
            return

        # ---------------- Finished ----------------

        if self.sample_count >= TARGET_SAMPLES:

            print("\nFinished!")

            self.recording = False

            self.current_label = None

            return

        # ---------------- One hand only ----------------

        if len(hands) != 1:
            return

        # ---------------- Save interval ----------------

        if time.time() - self.last_save < SAVE_INTERVAL:
            return

        hand = hands[0]

        # ---------------- Duplicate filter ----------------

        if self.previous_landmarks is not None:

            total = 0

            for old, new in zip(
                self.previous_landmarks,
                hand.landmarks,
            ):

                total += abs(old.x - new.x)
                total += abs(old.y - new.y)
                total += abs(old.z - new.z)

            average_difference = (
                total / (21 * 3)
            )

            if average_difference < 0.01:
                return

        # ---------------- Save ----------------

        self.writer.save(
            self.current_label,
            hand.landmarks,
        )

        self.previous_landmarks = hand.landmarks

        self.sample_count += 1

        self.last_save = time.time()

        print(
            f"\r{self.current_label.upper()} : "
            f"{self.sample_count}/{TARGET_SAMPLES}",
            end="",
        )

    # -------------------------------------------------

    def get_status(self):

        if self.current_label is None:
            return "Recorder: Idle"

        if self.countdown:

            remaining = max(
                0,
                3 - int(time.time() - self.countdown_start)
            )

            return f"Starting in {remaining}"

        if self.recording:

            return (
                f"Recording "
                f"{self.current_label.upper()} "
                f"{self.sample_count}/{TARGET_SAMPLES}"
            )

        return "Recorder: Idle"