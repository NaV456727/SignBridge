import cv2
import mediapipe as mp

from src.core.hand import Hand
from src.core.landmark import Landmark


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        detected_hands = []

        if not results.multi_hand_landmarks:
            return detected_hands

        image_height, image_width = frame.shape[:2]

        for landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):

            landmark_list = []
            x_list = []
            y_list = []

            for idx, landmark in enumerate(landmarks.landmark):

                # Pixel coordinates (for bounding box)
                pixel_x = int(landmark.x * image_width)
                pixel_y = int(landmark.y * image_height)

                x_list.append(pixel_x)
                y_list.append(pixel_y)

                # Store normalized coordinates
                landmark_list.append(
                    Landmark(
                        id=idx,
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z,
                    )
                )

            bbox = (
                min(x_list),
                min(y_list),
                max(x_list),
                max(y_list),
            )

            hand = Hand(
                landmarks=landmark_list,
                bbox=bbox,
                handedness=handedness.classification[0].label,
                confidence=handedness.classification[0].score,
            )

            detected_hands.append(hand)

        return detected_hands

    def draw(self, frame, hands):

        image_height, image_width = frame.shape[:2]

        for hand in hands:

            x1, y1, x2, y2 = hand.bbox

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # Label
            text = f"{hand.handedness} ({hand.confidence * 100:.0f}%)"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Draw landmarks
            for landmark in hand.landmarks:

                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 0, 255),
                    -1,
                )

        return frame