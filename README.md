# Hand Sign Translator — Vite

This is the browser version of the Python hand-sign project.

## 1. Copy your trained models

From your Python project, copy:

- `model/keypoint_classifier/keypoint_classifier.tflite`
- `model/point_history_classifier/point_history_classifier.tflite`

into:

```text
public/models/
```

Rename them exactly:

```text
keypoint_classifier.tflite
point_history_classifier.tflite
```

## 2. Copy your ASL images

Copy the contents of your Python project's `asl_images` folder into:

```text
public/asl_images/
```

The files should be named:

```text
A.jpg
B.jpg
...
Z.jpg
```

## 3. Install

Open the terminal in this project and run:

```powershell
npm install
```

## 4. Start

```powershell
npm run dev
```

Then open the local URL Vite gives you.

## Important

The JavaScript reproduces the important preprocessing from the Python app:

- 21 hand landmarks
- integer pixel coordinates
- landmark 0 as the origin
- flattening to 42 values
- division by the maximum absolute value

The keypoint TFLite model therefore receives `[1, 42]` float32 input and produces 36 class scores.

The point-history model receives `[1, 32]` float32 input and produces 4 class scores.

MediaPipe Hands is loaded from its official npm package assets through jsDelivr. The custom TFLite models remain local to the website.

The browser's Speech Recognition API and Speech Synthesis API are used for speech features.

## Included in this package

The two trained TFLite models from the uploaded project are already placed in
`public/models/`.

The ASL JPG images are not included because they were supplied inside a RAR
archive that cannot be extracted in this runtime. Copy `A.jpg` through `Z.jpg`
from your existing `asl_images` folder into `public/asl_images/`.
