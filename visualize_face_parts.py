from collections import OrderedDict
import argparse
from pathlib import Path

import cv2
import dlib
import numpy as np


FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17)),
])


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    overlay = image.copy()
    output = image.copy()

    if colors is None:
        colors = [
            (19, 199, 109),
            (79, 76, 240),
            (230, 159, 23),
            (168, 100, 168),
            (158, 163, 32),
            (163, 38, 32),
            (180, 42, 220),
        ]

    for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
        (j, k) = FACIAL_LANDMARKS_68_IDXS[name]
        pts = shape[j:k]

        if name == "jaw":
            for l in range(1, len(pts)):
                pt_a = tuple(pts[l - 1])
                pt_b = tuple(pts[l])
                cv2.line(overlay, pt_a, pt_b, colors[i], 2)
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Highlight facial regions using 68 landmarks.")
    parser.add_argument(
        "--shape-predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib's 68-point facial landmark model file.",
    )
    parser.add_argument(
        "--image",
        default="sample_images/3.jpg",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output",
        default="output_face_parts.jpg",
        help="Path to save the visualization.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=500,
        help="Resize input image to this width before detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    predictor_path = Path(args.shape_predictor)
    output_path = Path(args.output)

    if not predictor_path.exists():
        raise FileNotFoundError(
            f"Predictor not found: {predictor_path}. "
            "Download shape_predictor_68_face_landmarks.dat separately."
        )
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    height, width = image.shape[:2]
    scale = args.width / float(width)
    image = cv2.resize(image, (args.width, int(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detections = detector(gray, 1)
    if not detections:
        raise RuntimeError("No face detected in the image.")

    for rect in detections:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        output = visualize_facial_landmarks(image, shape)
        cv2.imwrite(str(output_path), output)
        print(f"Saved visualization to: {output_path}")
        return


if __name__ == "__main__":
    main()
