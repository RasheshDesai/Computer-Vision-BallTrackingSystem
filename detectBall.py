import cv2
import numpy as np
import os
import re
import shutil
import torch
from ultralytics import YOLO
import math

class YOLOBallTracking:
    def __init__(self, video_path, model_path, frame_dir='Frame', processed_frame_dir='Frame_b'):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.frame_dir = frame_dir
        self.processed_frame_dir = processed_frame_dir
        self.trajectory_points = []  # Will store the center points of the ball for speed calculations
        self.fps = None  # Frames per second of the video
        self.class_names = self.model.names  # Get class names from YOLO model
        self.ball_speed = None  # Ball speed in pixels per second
        self.prev_center = None  # To store the previous center of the ball

    def cleanup_directories(self, directories):
        """Remove and recreate frame directories."""
        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)

    def process_video(self):
        """Process video frames to track ball and calculate speed."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        # Set FPS from video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.cleanup_directories([self.frame_dir, self.processed_frame_dir])

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = f'{self.frame_dir}/{frame_count}.png'
            cv2.imwrite(frame_path, frame)

            # Use YOLO to get object detection results
            results = self.model(frame)[0]
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(cls)  # Get the class ID
                label = self.class_names[class_id]  # Map ID to class name

                # Define colors for different objects
                color = (0, 255, 0) if label.lower() == "player" else (0, 0, 255)  # Green for Player, Red for Ball
                
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # If the detected object is a ball, calculate its speed
                if label.lower() == "ball":
                    # Calculate center of the ball
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    self.trajectory_points.append(center)
                    
                    # Update previous ball center for the next frame
                    self.prev_center = center

            # Save the processed frame
            processed_path = f'{self.processed_frame_dir}/processed_{frame_count}.png'
            cv2.imwrite(processed_path, frame)
            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def create_output_video(self, output_filename='output_video.mp4'):
        """Create a video from the processed frames."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frames = sorted(os.listdir(self.processed_frame_dir), key=lambda x: int(re.sub('\\D', '', x)))

        first_frame = cv2.imread(os.path.join(self.processed_frame_dir, frames[0]))
        height, width, _ = first_frame.shape
        size = (width, height)

        # Initialize video writer and write processed frames to video
        out = cv2.VideoWriter(output_filename, fourcc, self.fps, size)
        for frame_filename in frames:
            frame = cv2.imread(os.path.join(self.processed_frame_dir, frame_filename))
            out.write(frame)

        out.release()
        print(f"Video saved as {output_filename}")

if __name__ == "__main__":
    video_path = 'WhiteBall.mp4'  # Path to the input video file
    model_path = 'yolov8n.pt'  # Path to the YOLO model file (change as necessary)
    tracker = YOLOBallTracking(video_path, model_path)
    tracker.process_video()  
    tracker.create_output_video()  