from ultralytics import YOLO
import cv2
import os

model_path = os.path.join('runs','detect','train8','weights','best.pt')
model = YOLO(model_path)
model.export(format='onnx')