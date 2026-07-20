import csv
from pathlib import Path


class CSVWriter:

    def __init__(self, filepath):
        self.filepath = Path(filepath)

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        if not self.filepath.exists():
            self.create_file()

    def create_file(self):

        header = ["label"]

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}",
            ])

        with open(self.filepath, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)

    def save(self, label, landmarks):

        row = [label]

        for landmark in landmarks:
            row.extend([
                landmark.x,
                landmark.y,
                landmark.z,
            ])

        with open(self.filepath, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)