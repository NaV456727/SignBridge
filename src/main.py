import cv2
import time

from src.camera.webcam import Webcam
from src.detector.hand_detector import HandDetector
from src.training.dataset_recorder import DatasetRecorder
from src.inference.gesture_classifier import GestureClassifier
from src.ui.overlay import Overlay
from src.sentence.sentence_builder import SentenceBuilder


def main():

    webcam = Webcam()
    detector = HandDetector()
    classifier = GestureClassifier()
    recorder = DatasetRecorder()
    overlay = Overlay()
    builder = SentenceBuilder()

    prev_time = time.time()

    while True:

        frame = webcam.read_frame()

        if frame is None:
            break

        frame = cv2.flip(frame, 1)

        hands = detector.detect(frame)

        confidence = 0
        top_predictions = []

        if len(hands) == 1:

            raw_prediction, confidence, top_predictions = classifier.predict(hands[0])

            builder.update(raw_prediction, confidence)

        else:
            top_predictions = []
            builder.update("", 0)

        # FIX: get_prediction() can return None (e.g. before the
        # sentence builder has locked in a stable letter). Both
        # branches now fall back to "" so overlay never receives None.
        prediction = builder.get_prediction() or ""

        frame = detector.draw(frame, hands)

        status = recorder.get_status()

        current_time = time.time()

        # FIX: guard against a zero (or negative, on clock adjustments)
        # elapsed time, which previously could raise ZeroDivisionError.
        elapsed = current_time - prev_time
        fps = int(1 / elapsed) if elapsed > 0 else 0

        prev_time = current_time

        frame = overlay.draw(
            frame=frame,
            prediction=prediction,
            confidence=confidence,
            sentence=builder.get_sentence(),
            fps=fps,
            hand_count=len(hands),
            mode="ASL",
            top_predictions=top_predictions,
        )

        cv2.imshow("SignBridge", frame)

        key = cv2.waitKey(1) & 0xFF

        if key != 255:

            recorder.process_key(key)

        recorder.update(hands)

        if key == ord("q"):
            break

    webcam.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()