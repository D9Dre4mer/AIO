"""
Object detection and crop for action recognition preprocessing.
Backend: YOLO (Ultralytics) — YOLO11, YOLO26, ... — hoặc torchvision Faster R-CNN.
Dùng object detection để khoanh vùng vật thể/chủ thể, cắt frame focus.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch

logger = logging.getLogger(__name__)

# Windows: đường dẫn dài gây [Errno 22] Invalid argument
MAX_PATH_WIN = 200

# Windows: O_BINARY để đọc ảnh đúng
O_BINARY = getattr(os, "O_BINARY", 0)


def _load_image_from_path(path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Đọc ảnh từ path. Trên Windows path dài: dùng \\?\ hoặc copy qua file tạm (đường dẫn ngắn).
    Trả về (array HWC RGB uint8, None) nếu OK; (None, reason_str) nếu lỗi (reason để log).
    """
    try:
        s = str(path.resolve())
    except OSError:
        try:
            s = str(path.absolute())
        except OSError:
            s = str(Path.cwd() / path)
    use_long = sys.platform == "win32" and len(s) >= MAX_PATH_WIN
    path_str = ("\\\\?\\" + s) if (use_long and not s.startswith("\\\\?\\")) else s

    def open_and_load(path_or_fd):
        if isinstance(path_or_fd, int):
            f = os.fdopen(path_or_fd, "rb")
        else:
            f = open(path_or_fd, "rb")
        try:
            with Image.open(f) as im:
                img = im.convert("RGB")
            return np.array(img)
        finally:
            f.close()

    try:
        with open(path_str, "rb") as f:
            with Image.open(f) as im:
                img = im.convert("RGB")
        return (np.array(img), None)
    except OSError as e1:
        pass
    if sys.platform != "win32":
        return (None, "step1 failed errno=%s (not Windows)" % getattr(e1, "errno", "?"))
    try:
        fd = os.open(path_str, os.O_RDONLY | O_BINARY)
        arr = open_and_load(fd)
        return (arr, None)
    except OSError:
        pass
    tmp_path = None
    try:
        fd = os.open(path_str, os.O_RDONLY | O_BINARY)
        f = os.fdopen(fd, "rb")
        data = f.read()
        f.close()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=path.suffix)
        tmp_path = tmp.name
        tmp.write(data)
        tmp.close()
        arr = open_and_load(tmp_path)
        return (arr, None)
    except OSError as e3:
        return (None, "step3 temp copy+PIL errno=%s" % getattr(e3, "errno", "?"))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

# COCO torchvision: 0 = background, 1 = person
COCO_PERSON_LABEL = 1
# Ultralytics COCO: class 0 = person (0-indexed)
YOLO_PERSON_CLASS_ID = 0


def _get_yolo_model(model_name: str = "yolo11m.pt"):
    """Load YOLO (Ultralytics): YOLO11, YOLO26, ... Cần: pip install ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Backend yolo cần ultralytics. Chạy: pip install ultralytics"
        )
    return YOLO(model_name)


def _get_torchvision_model(device: torch.device):
    """Load Faster R-CNN pretrained on COCO (chỉ dùng khi cần)."""
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    except ImportError:
        weights = "DEFAULT"
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model.to(device)


def detect_object_boxes_yolo(
    image: np.ndarray,
    model,
    score_threshold: float = 0.5,
    person_only: bool = False,
    person_first: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    Trả về danh sách (x1, y1, x2, y2) từ YOLO (YOLO11, YOLO26, ...). image: HWC RGB uint8.
    Ultralytics COCO: class 0 = person.
    """
    results = model(image, verbose=False)
    if not results or not results[0].boxes:
        return []
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    candidates = []
    for i in range(len(cls)):
        if conf[i] < score_threshold:
            continue
        lab = int(cls[i])
        if person_only and lab != YOLO_PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = xyxy[i]
        candidates.append(((int(x1), int(y1), int(x2), int(y2)), lab))
    if not candidates:
        return []
    if person_first:
        person_boxes = [
            box for box, lab in candidates if lab == YOLO_PERSON_CLASS_ID
        ]
        if person_boxes:
            return person_boxes
    return [box for box, _ in candidates]


def detect_object_boxes(
    image_tensor: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    score_threshold: float = 0.5,
    person_only: bool = False,
    person_first: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    Trả về danh sách (x1, y1, x2, y2) từ Faster R-CNN (torchvision).
    person_first: nếu có người thì chỉ lấy box người; không có người mới lấy vật thể khác.
    """
    with torch.no_grad():
        preds = model([image_tensor.to(device)])
    boxes = preds[0]["boxes"].cpu()
    labels = preds[0]["labels"].cpu()
    scores = preds[0]["scores"].cpu()
    candidates = []
    for i in range(len(labels)):
        lab = int(labels[i].item())
        if lab < 1 or scores[i].item() < score_threshold:
            continue
        if person_only and lab != COCO_PERSON_LABEL:
            continue
        b = boxes[i].tolist()
        candidates.append(((int(b[0]), int(b[1]), int(b[2]), int(b[3])), lab))
    if not candidates:
        return []
    if person_first:
        person_boxes = [box for box, lab in candidates if lab == COCO_PERSON_LABEL]
        if person_boxes:
            return person_boxes
    return [box for box, _ in candidates]


def expand_bbox(
    x1: int, y1: int, x2: int, y2: int,
    height: int, width: int,
    padding_ratio: float = 0.2,
    min_crop_side: int = 64,
) -> Tuple[int, int, int, int]:
    """Mở rộng bbox với padding, clip trong ảnh, đảm bảo kích thước tối thiểu."""
    w, h = x2 - x1, y2 - y1
    pad_w = max(min_crop_side, int(w * padding_ratio))
    pad_h = max(min_crop_side, int(h * padding_ratio))
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(width, x2 + pad_w)
    y2 = min(height, y2 + pad_h)
    return (x1, y1, x2, y2)


def merge_boxes(
    boxes: List[Tuple[int, int, int, int]],
    height: int,
    width: int,
    padding_ratio: float = 0.15,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Gộp nhiều box thành một vùng (union), mở rộng padding, clip.
    Trả về (x1, y1, x2, y2) hoặc None nếu không có box.
    """
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return expand_bbox(x1, y1, x2, y2, height, width, padding_ratio=padding_ratio)


def crop_to_region(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    padding_ratio: float = 0.15,
    fallback_full_frame: bool = True,
) -> np.ndarray:
    """
    Cắt ảnh về vùng vật thể (union box + padding).
    image: HWC, uint8.
    Nếu không có box và fallback_full_frame=True thì trả về ảnh gốc.
    """
    h, w = image.shape[:2]
    region = merge_boxes(boxes, h, w, padding_ratio=padding_ratio)
    if region is None:
        return image if fallback_full_frame else np.zeros_like(image)
    x1, y1, x2, y2 = region
    return np.ascontiguousarray(image[y1:y2, x1:x2])


class PersonFocusCrop:
    """
    Preprocessing: load frame -> detect object (or person only) -> crop to region.
    backend: "yolo11" (mặc định, Ultralytics) hoặc "torchvision" (Faster R-CNN).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        score_threshold: float = 0.5,
        padding_ratio: float = 0.15,
        fallback_full_frame: bool = True,
        person_only: bool = False,
        person_first: bool = True,
        backend: str = "yolo11",
        yolo_model: str = "yolo26m.pt",
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.score_threshold = score_threshold
        self.padding_ratio = padding_ratio
        self.fallback_full_frame = fallback_full_frame
        self.person_only = person_only
        self.person_first = person_first
        self.backend = backend.lower()
        self.yolo_model = yolo_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if self.backend == "yolo11":
                self._model = _get_yolo_model(self.yolo_model)
            else:
                self._model = _get_torchvision_model(self.device)
        return self._model

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Chuẩn hóa ảnh cho Faster R-CNN: [C,H,W] float, ImageNet normalize."""
        from torchvision import transforms
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        return t(img)

    def process_frame(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Nhận ảnh HWC uint8 RGB, trả về (ảnh đã cắt vùng vật thể hoặc None, had_object).
        """
        if self.backend == "yolo11":
            boxes = detect_object_boxes_yolo(
                image,
                self.model,
                score_threshold=self.score_threshold,
                person_only=self.person_only,
                person_first=self.person_first,
            )
        else:
            tensor = self._preprocess_image(image)
            boxes = detect_object_boxes(
                tensor,
                self.model,
                self.device,
                score_threshold=self.score_threshold,
                person_only=self.person_only,
                person_first=self.person_first,
            )
        if not boxes and not self.fallback_full_frame:
            return (None, False)
        out = crop_to_region(
            image,
            boxes,
            padding_ratio=self.padding_ratio,
            fallback_full_frame=self.fallback_full_frame,
        )
        return (out, True)

    def process_frame_path(self, path: Path) -> Tuple[Optional[np.ndarray], bool]:
        """Đọc ảnh từ path, detect + crop. Trả về (ảnh, had_object); (None, False) nếu lỗi hoặc không có vật thể."""
        try:
            arr, load_reason = _load_image_from_path(path)
            if arr is None:
                logger.warning("Process frame %s failed: could not load image (%s)", path, load_reason or "unknown")
                return (None, False)
            return self.process_frame(arr)
        except Exception as e:
            logger.warning("Process frame %s failed: %s", path, e)
            return (None, False)
