import sys
from pathlib import Path

from src.config import DATASET_FILE
from src.converter.image_processor import ImageProcessor
from src.converter.csv_exporter import CSVExporter


class DatasetConverter:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.dataset_name = self.dataset_path.name

        self.processor = ImageProcessor()

        self.exporter = CSVExporter("dataset/asl_dataset.csv")

        self.saved = 0
        self.skipped = 0

    def convert(self):

        print("=" * 60)
        print(f"Dataset : {self.dataset_name}")
        print("=" * 60)

        # Every gesture folder
        for label_folder in sorted(self.dataset_path.iterdir()):

            if not label_folder.is_dir():
                continue

            label = label_folder.name

            images = []

            for extension in (
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.bmp",
                "*.JPG",
                "*.JPEG",
                "*.PNG"
            ):
                images.extend(label_folder.glob(extension))

            total = len(images)

            print(f"\nProcessing '{label}' ({total} images)")

            for index, image_path in enumerate(images, start=1):

                print(
                    f"\r{index}/{total}",
                    end="",
                    flush=True,
                )

                hand = self.processor.process(str(image_path))

                if hand is None:

                    self.skipped += 1
                    continue

                self.exporter.save(
                    self.dataset_name,
                    image_path.name,
                    label,
                    hand.landmarks,
                )

                self.saved += 1

            print()

        print("\n" + "=" * 60)

        print(f"Saved   : {self.saved}")
        print(f"Skipped : {self.skipped}")

        print("=" * 60)


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "python src/converter/dataset_converter.py datasets/asl_alphabet"
        )

        return

    dataset = sys.argv[1]

    converter = DatasetConverter(dataset)

    converter.convert()


if __name__ == "__main__":
    main()