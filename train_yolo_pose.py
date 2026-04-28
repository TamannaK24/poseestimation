from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_DIR = Path(r"C:\Projects\PoseEstimation")

JSON_DIR = PROJECT_DIR / "auto_pose_output" / "json_labels"
DATASET_DIR = PROJECT_DIR / "dataset_pose"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# For deciding whether an auto-labeled keypoint counts as visible
KPT_CONF_THRESHOLD = 0.20

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def make_dirs(base: Path) -> None:
    for split in ["train", "val", "test"]:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def xyxy_to_xywhn(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return clamp01(cx), clamp01(cy), clamp01(w), clamp01(h)


def safe_name_from_json(json_path: Path) -> str:
    return json_path.stem


def build_yolo_pose_line(det: dict, img_w: int, img_h: int) -> str | None:
    """Convert one detection dict from the JSON output into one YOLO pose label line."""
    bbox = det.get("bbox_xyxy")

    if not bbox or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = bbox
    cx, cy, w, h = xyxy_to_xywhn(x1, y1, x2, y2, img_w, img_h)

    # class 0 = person
    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

    keypoints = det.get("keypoints", {})

    for name in KEYPOINT_NAMES:
        kp = keypoints.get(name)

        if kp is None:
            # missing keypoint
            parts.extend(["0.000000", "0.000000", "0"])
            continue

        x = kp.get("x", 0.0)
        y = kp.get("y", 0.0)
        conf = kp.get("confidence", None)

        # Normalize x, y
        xn = clamp01(float(x) / img_w)
        yn = clamp01(float(y) / img_h)

        # Visibility flag: 2 = visible/labeled, 0 = not labeled
        if conf is None:
            v = 2
        elif float(conf) >= KPT_CONF_THRESHOLD:
            v = 2
        else:
            v = 0

        parts.extend([f"{xn:.6f}", f"{yn:.6f}", str(v)])

    return " ".join(parts)


def process_split(files: list[Path], split: str) -> int:
    count = 0

    for json_path in files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = Path(data["image_path"])

        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        detections = data.get("detections", [])

        if not detections:
            # No detections -> skip for pose training
            continue

        # Use original image
        img = Image.open(image_path)
        img_w, img_h = img.size

        # Build label lines, one per detected person
        label_lines = []
        for det in detections:
            line = build_yolo_pose_line(det, img_w, img_h)
            if line is not None:
                label_lines.append(line)

        if not label_lines:
            continue

        stem = safe_name_from_json(json_path)
        out_img_path = DATASET_DIR / "images" / split / f"{stem}{image_path.suffix.lower()}"
        out_lbl_path = DATASET_DIR / "labels" / split / f"{stem}.txt"

        shutil.copy2(image_path, out_img_path)

        with open(out_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")

        count += 1

    return count


def write_data_yaml() -> Path:
    yaml_text = f"""
path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
""".strip()

    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("Saved:", yaml_path)
    print(yaml_text)

    return yaml_path


def print_sanity_check() -> None:
    for split in ["train", "val", "test"]:
        img_count = len(list((DATASET_DIR / "images" / split).glob("*")))
        lbl_count = len(list((DATASET_DIR / "labels" / split).glob("*.txt")))
        print(split, "images:", img_count, "labels:", lbl_count)


def build_dataset() -> Path:
    json_files = sorted(JSON_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON label files.")

    if not json_files:
        raise ValueError("No JSON files found. Check JSON_DIR.")

    random.seed(RANDOM_SEED)
    json_files_shuffled = json_files[:]
    random.shuffle(json_files_shuffled)

    n = len(json_files_shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = json_files_shuffled[:n_train]
    val_files = json_files_shuffled[n_train : n_train + n_val]
    test_files = json_files_shuffled[n_train + n_val :]

    print("Split sizes:")
    print("train:", len(train_files))
    print("val: ", len(val_files))
    print("test: ", len(test_files))

    make_dirs(DATASET_DIR)

    train_count = process_split(train_files, "train")
    val_count = process_split(val_files, "val")
    test_count = process_split(test_files, "test")

    print("Actually written:")
    print("train:", train_count)
    print("val: ", val_count)
    print("test: ", test_count)

    yaml_path = write_data_yaml()
    print_sanity_check()

    return yaml_path


def train_model(yaml_path: Path) -> YOLO:
    model = YOLO("yolov8n-pose.pt")

    model.train(
        data=str(yaml_path),
        epochs=30,
        imgsz=640,
        batch=16,
        device="cpu",
        project=str(PROJECT_DIR / "runs_pose"),
        name="yolov8n_pose_finetune",
        pretrained=True,
    )

    return model


def evaluate_model(model: YOLO, yaml_path: Path) -> None:
    # Validate on val set
    metrics = model.val(data=str(yaml_path), split="val")
    print("Validation metrics:")
    print(metrics)

    # Test on test set
    test_metrics = model.val(data=str(yaml_path), split="test")
    print("Test metrics:")
    print(test_metrics)


def save_sample_predictions() -> None:
    best_model_path = PROJECT_DIR / "runs_pose" / "yolov8n_pose_finetune" / "weights" / "best.pt"
    best_model = YOLO(str(best_model_path))

    sample_test_images = list((DATASET_DIR / "images" / "test").glob("*"))[:5]
    print("Sample test images:")
    for img_path in sample_test_images:
        print(img_path)

    pred_dir = PROJECT_DIR / "sample_predictions"
    pred_dir.mkdir(exist_ok=True)

    for img_path in sample_test_images:
        res = best_model.predict(source=str(img_path), conf=0.25, save=False, verbose=False)
        plotted = res[0].plot()
        out_path = pred_dir / img_path.name
        cv2.imwrite(str(out_path), plotted)
        print(f"Saved prediction: {out_path}")


def main() -> None:
    yaml_path = build_dataset()
    model = train_model(yaml_path)
    evaluate_model(model, yaml_path)
    save_sample_predictions()


if __name__ == "__main__":
    main()
