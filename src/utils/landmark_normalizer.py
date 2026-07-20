import math
import pandas as pd


def normalize_features(features):
    """
    Normalize 21 hand landmarks.

    Steps:
    1. Translate wrist to the origin.
    2. Scale the hand using the maximum wrist distance.
    """

    # ---------------------------------
    # Convert flat list to points
    # ---------------------------------

    points = []

    for i in range(0, len(features), 3):

        points.append([
            features[i],
            features[i + 1],
            features[i + 2],
        ])

    # ---------------------------------
    # Wrist
    # ---------------------------------

    wrist_x, wrist_y, wrist_z = points[0]

    translated = []

    for x, y, z in points:

        translated.append([
            x - wrist_x,
            y - wrist_y,
            z - wrist_z,
        ])

    # ---------------------------------
    # Compute scale
    # Largest distance from wrist
    # ---------------------------------

    max_distance = 0.0

    for x, y, z in translated:

        distance = math.sqrt(
            x * x +
            y * y +
            z * z
        )

        if distance > max_distance:
            max_distance = distance

    if max_distance == 0:
        max_distance = 1.0

    # ---------------------------------
    # Normalize
    # ---------------------------------

    normalized = []

    for x, y, z in translated:

        normalized.extend([
            x / max_distance,
            y / max_distance,
            z / max_distance,
        ])

    return normalized


def normalize_dataframe(df):

    normalized_rows = []

    for _, row in df.iterrows():

        normalized_rows.append(
            normalize_features(row.tolist())
        )

    return pd.DataFrame(
        normalized_rows,
        columns=df.columns,
    )