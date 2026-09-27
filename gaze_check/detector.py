"""Телефон и человек в кадре: MediaPipe ObjectDetector (EfficientDet-Lite, COCO) на процессоре.

Тот же MediaPipe, что и для глаз: никаких лишних библиотек, модель — несколько мегабайт,
десятки миллисекунд на кадр."""
from .workflow import PERSON_CLASS, PERSON_CONFIDENCE, PHONE_CLASS

PHONE_CONFIDENCE = 0.3  # у EfficientDet уверенность ниже, чем у RF-DETR: 0.3 здесь ≈ 0.4 там
COCO_ID = {PERSON_CLASS: 1, PHONE_CLASS: 77}  # номера категорий COCO, как были у RF-DETR


class Detector:
    def __init__(self, model_path, confidence=PHONE_CONFIDENCE):
        from .mp import mediapipe
        self.mp, vision, BaseOptions = mediapipe()
        self.confidence = confidence
        opts = vision.ObjectDetectorOptions(
            # байтами, а не путём: MediaPipe не открывает пути с кириллицей
            base_options=BaseOptions(model_asset_buffer=open(model_path, "rb").read()),
            running_mode=vision.RunningMode.IMAGE, max_results=10,
            score_threshold=min(confidence, PERSON_CONFIDENCE),
            category_allowlist=[PERSON_CLASS, PHONE_CLASS])
        self.model = vision.ObjectDetector.create_from_options(opts)

    def detect(self, rgb):
        """RGB-кадр → [{"x", "y" (центр), "width", "height", "confidence", "class_id", "class"}]."""
        mp = self.mp
        res = self.model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        out = []
        for d in res.detections:
            c = d.categories[0]
            need = self.confidence if c.category_name == PHONE_CLASS else PERSON_CONFIDENCE
            if c.score < need:
                continue
            b = d.bounding_box
            out.append({"x": b.origin_x + b.width / 2, "y": b.origin_y + b.height / 2,
                        "width": float(b.width), "height": float(b.height), "confidence": float(c.score),
                        "class_id": COCO_ID[c.category_name], "class": c.category_name})
        return out
