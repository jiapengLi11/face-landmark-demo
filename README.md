# Face Landmark Demo

This is a small practice project for face landmark detection and facial region visualization using `dlib` and `OpenCV`.

## Included

- `landmark_points.py`: detect a face and draw the 68 landmark points
- `visualize_face_parts.py`: highlight facial regions such as eyes, eyebrows, nose, mouth, and jaw
- `sample_images/`: example images used for local testing
- `figures/`: screenshots from earlier experiments

## Not Included

The repository does **not** include `shape_predictor_68_face_landmarks.dat`.

That file is large and should be downloaded separately from dlib / iBUG resources.

## Setup

```bash
pip install -r requirements.txt
```

Download the dlib predictor file and place it in the project root:

```text
shape_predictor_68_face_landmarks.dat
```

## Run

Draw landmark points:

```bash
python landmark_points.py --shape-predictor shape_predictor_68_face_landmarks.dat --image sample_images/3.jpg
```

Visualize facial regions:

```bash
python visualize_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image sample_images/3.jpg
```

## Notes

- This project is a cleaned version of an older local practice folder.
- The focus is demonstration and visualization, not training a new landmark model.
