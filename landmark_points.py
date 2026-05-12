from collections import OrderedDict
import argparse
from pathlib import Path

import cv2
import dlib


FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17)),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw 68 facial landmarks on an image.")
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
        default="output_landmarks.jpg",
        help="Path to save the annotated image.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not render the landmark index labels.",
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

    detections = detector(image, 1)
    if not detections:
        raise RuntimeError("No face detected in the image.")

    for face_index, rect in enumerate(detections):
        shape = predictor(image, rect)
        print(
            f"Face {face_index}: "
            f"left={rect.left()} top={rect.top()} right={rect.right()} bottom={rect.bottom()}"
        )
        for index, point in enumerate(shape.parts(), start=1):
            point_pos = (point.x, point.y)
            cv2.circle(image, point_pos, 1, (255, 0, 0), 2)
            if not args.no_labels:
                cv2.putText(
                    image,
                    str(index),
                    point_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

    cv2.imwrite(str(output_path), image)
    print(f"Saved annotated image to: {output_path}")


if __name__ == "__main__":
    main()
