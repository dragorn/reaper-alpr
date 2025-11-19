## 🚀 Quick Start

Here's how to get started with FastALPR:

### Predictions

```python
from fast_alpr import ALPR

# You can also initialize the ALPR with custom plate detection and OCR models.
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model",
)

# The "assets/test_image.png" can be found in repo root dir
# You can also pass a NumPy array containing cropped plate image
alpr_results = alpr.predict("assets/test_image.png")
print(alpr_results)
```

???+ note

    See [reference](reference.md) for the available models.

Output:

<img alt="ALPR Result" height="350" src="https://raw.githubusercontent.com/ankandrew/fast-alpr/5063bd92fdd30f46b330d051468be267d4442c9b/assets/alpr_result.webp" width="700"/>

### Draw Results

You can also **draw** the predictions directly on the image:

```python
import cv2

from fast_alpr import ALPR

# Initialize the ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model",
)

# Load the image
image_path = "assets/test_image.png"
frame = cv2.imread(image_path)

# Draw predictions on the image
annotated_frame = alpr.draw_predictions(frame)
```

Annotated frame:

<img alt="ALPR Draw Predictions" src="https://raw.githubusercontent.com/ankandrew/fast-alpr/0a6076dcb8d9084514fe47e8abaaeb77cae45f8e/assets/alpr_draw_predictions.png"/>

