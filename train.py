from ultralytics import YOLO
import torch
import os

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "yolov12n.pt"

DATA_CONFIG = "/content/drive/MyDrive/ai defect/Industrial-defect-detection-1/data.yaml"

PROJECT_PATH = "/content/drive/MyDrive/ai defect/runs"

MODEL_NAME = "industrial_detector"

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16

# =========================================================
# GPU CHECK
# =========================================================

if torch.cuda.is_available():

    DEVICE = 0

    print("GPU:", torch.cuda.get_device_name(0))

else:

    DEVICE = "cpu"

    print("Using CPU")

# =========================================================
# CHECK FILES
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not os.path.exists(DATA_CONFIG):
    raise FileNotFoundError(
        f"Data config not found: {DATA_CONFIG}"
    )

# =========================================================
# LOAD MODEL
# =========================================================

model = YOLO(MODEL_PATH)

# =========================================================
# TRAIN
# =========================================================

results = model.train(

    data=DATA_CONFIG,

    epochs=50,

    imgsz=640,

    batch=16,

    device=DEVICE,

    workers=2,

    cache=True,

    pretrained=True,

    augment=True,

    optimizer="AdamW",

    lr0=0.001,

    patience=20,

    project=PROJECT_PATH,

    name=MODEL_NAME,

    exist_ok=True,

    val=True,

    plots=True,

    save=True,

    amp=True
)

# =========================================================
# VALIDATION
# =========================================================

metrics = model.val()

print("\n========== INDUSTRIAL RESULTS ==========\n")

print("mAP50:", metrics.box.map50)

print("mAP50-95:", metrics.box.map)

# =========================================================
# EXPORT
# =========================================================

model.export(
    format="onnx",
    simplify=True
)

print("\nIndustrial Model Training Complete!")

print(
    f"\nBest Model:\n{PROJECT_PATH}/{MODEL_NAME}/weights/best.pt"
)