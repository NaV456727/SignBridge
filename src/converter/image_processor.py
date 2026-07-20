import cv2

from src.detector.hand_detector import HandDetector


class ImageProcessor:

    def __init__(self):

        self.detector = HandDetector()

    def process(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            return None

        hands = self.detector.detect(image)

        if len(hands) != 1:
            return None

        return hands[0]