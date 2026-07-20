from collections import deque, Counter
import time


class SentenceBuilder:

    def __init__(self):

        self.sentence = ""

        self.history = deque(maxlen=10)

        self.stable_prediction = None

        self.last_committed = None

        self.ready_for_new_letter = True

        self.last_commit_time = 0

        self.cooldown = 0.35

        self.min_confidence = 40

        self.required_votes = 8

    def update(self, prediction, confidence):

        # ---------------- No hand ----------------

        if prediction == "":

            self.history.clear()

            self.stable_prediction = None

            self.ready_for_new_letter = True

            return

        # ---------------- Low confidence ----------------

        if confidence < self.min_confidence:
            return

        # ---------------- History ----------------

        self.history.append(prediction)

        if len(self.history) < self.history.maxlen:
            return

        counts = Counter(self.history)

        stable_prediction, votes = counts.most_common(1)[0]

        if votes < self.required_votes:
            return

        self.stable_prediction = stable_prediction

        # ---------------- Cooldown ----------------

        if time.time() - self.last_commit_time < self.cooldown:
            return

        # ---------------- Already committed ----------------

        if (
            stable_prediction == self.last_committed
            and not self.ready_for_new_letter
        ):
            return

        # ---------------- Commit ----------------

        if stable_prediction == "space":

            self.sentence += " "

        elif stable_prediction == "del":

            self.sentence = self.sentence[:-1]

        else:

            self.sentence += stable_prediction

        self.last_committed = stable_prediction

        self.ready_for_new_letter = False

        self.last_commit_time = time.time()

        print("Sentence:", self.sentence)

    def get_prediction(self):

        return self.stable_prediction

    def get_sentence(self):

        return self.sentence

    def clear(self):

        self.sentence = ""

        self.history.clear()

        self.last_committed = None

        self.ready_for_new_letter = True