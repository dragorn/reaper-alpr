## 🛠️ Customization and Flexibility

FastALPR is designed to be flexible. You can customize the detector and OCR models according to your requirements.

### Using Tesseract OCR

You can very easily integrate with **Tesseract** OCR to leverage its capabilities:

```python title="tesseract_ocr.py"
import re
from statistics import mean

import numpy as np
import pytesseract

from fast_alpr.alpr import ALPR, BaseOCR, OcrResult


class PytesseractOCR(BaseOCR):
    def __init__(self) -> None:
        """
        Init PytesseractOCR.
        """

    def predict(self, cropped_plate: np.ndarray) -> OcrResult | None:
        if cropped_plate is None:
            return None
        # You can change 'eng' to the appropriate language code as needed
        data = pytesseract.image_to_data(
            cropped_plate,
            lang="eng",
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
        )
        plate_text = " ".join(data["text"]).strip()
        plate_text = re.sub(r"[^A-Za-z0-9]", "", plate_text)
        avg_confidence = mean(conf for conf in data["conf"] if conf > 0) / 100.0
        return OcrResult(text=plate_text, confidence=avg_confidence)


alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end", ocr=PytesseractOCR())

alpr_results = alpr.predict("assets/test_image.png")
print(alpr_results)
```

???+ tip

    You can implement this with any OCR you want! For example, [EasyOCR](https://github.com/JaidedAI/EasyOCR).
