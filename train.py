from ultralytics import YOLO
from ultralytics import settings
from collections import OrderedDict
import torch

# Update a setting
settings.update({"runs_dir": "./runs"})

model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
results = model.train(data="./datasets/brain-tumor-detection/BrainTumorYolov8_subset/data.yaml", epochs=5, imgsz=640)