from ultralytics import YOLO
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

model = YOLO("yolov8s.pt")
results = model.train(data="data.yaml", epochs=100)