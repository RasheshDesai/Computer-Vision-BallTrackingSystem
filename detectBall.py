import cv2
import numpy as np
import os
import re
import shutil
from ultralytics import YOLO  # Import YOLO
from inference_sdk import InferenceHTTPClient #For stunps model

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Mj7AnsJsz7fcORL4eZ7B"
)

class BallTracking:
    def __init__(self, video_path, model_path, frame_dir='Frame', processed_frame_dir='Frame_b'):
        self.video_path = video_path
        self.model = YOLO(model_path)  # Load the YOLO model
        self.frame_dir = frame_dir
        self.processed_frame_dir = processed_frame_dir
        self.trajectory_points = []
        self.fps = None
        self.time_between_frames = None
        self.bounce_points = []

    @staticmethod
    def calculate_displacement(point1, point2, pixel_to_distance_ratio):
        dx = (point2[0] - point1[0]) * pixel_to_distance_ratio
        dy = (point2[1] - point1[1]) * pixel_to_distance_ratio
        return (dx**2 + dy**2)**0.5

    @staticmethod
    def cleanup_directories(directories):
        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.time_between_frames = 1 / self.fps

        self.cleanup_directories([self.frame_dir, self.processed_frame_dir])

        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            roi = frame
            frame_path = f'{self.frame_dir}/{cnt}.png'
            cv2.imwrite(frame_path, roi)

            # Use YOLO to detect the ball
            results = self.model(frame)
            detected_balls = []

            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    radius = max((x2 - x1) // 2, (y2 - y1) // 2)
                    detected_balls.append((center, radius))

            if detected_balls:
                largest_ball = max(detected_balls, key=lambda x: x[1])  # Select largest detected ball
                largest_center, largest_radius = largest_ball

                self.trajectory_points.append(largest_center)
                cv2.circle(frame, largest_center, largest_radius, (0, 255, 0), 2)  # Green circle around ball

            # COMMENTED OUT STUMPS DETECTION
            # Use Inference Client to detect stumps
            result = CLIENT.infer(frame_path, model_id="cricket-9czj5/3")
            detected_stumps = []
            for prediction in result["predictions"]:
                x, y, width, height = int(prediction["x"]), int(prediction["y"]), int(prediction["width"]), int(prediction["height"])
                detected_stumps.append((x, y, width, height))

            if detected_stumps:
                self.last_stump_position = detected_stumps  # Store last detected stump position
                for x, y, width, height in detected_stumps:
                    cv2.rectangle(frame, (x - width // 2, y - height // 2), (x + width // 2, y + height // 2), (0, 0, 255), 2)  # Red box around stumps
            elif self.last_stump_position:
                for x, y, width, height in self.last_stump_position:
                    cv2.rectangle(frame, (x - width // 2, y - height // 2), (x + width // 2, y + height // 2), (255, 0, 0), 2)  # Blue box for last known stumps

            cv2.imwrite(f'{self.processed_frame_dir}/processed_frame_{cnt}.png', frame)
            cnt += 1

            self.detect_bounces(frame)

        cap.release()
        cv2.destroyAllWindows()

    def detect_bounces(self, frame):
        if len(self.trajectory_points) < 3:
            return

        for i in range(1, len(self.trajectory_points) - 1):
            prev_y = self.trajectory_points[i - 1][1]
            current_y = self.trajectory_points[i][1]
            next_y = self.trajectory_points[i + 1][1]

            if prev_y > current_y < next_y:  # Local minimum in y-axis (bounce point)
                self.bounce_points.append(self.trajectory_points[i])
                print(f"Bounce detected at: {self.trajectory_points[i]}")

                # Mark the bounce point with a red circle on the current frame
                cv2.circle(frame, self.trajectory_points[i], 5, (0, 0, 255), -1)  # Red dot at bounce location

                # You can save or further process this frame if needed

    def calculate_speeds(self):
        if len(self.trajectory_points) > 1:
            pixel_to_distance_ratio = 0.01
            speeds = []

            for i in range(1, len(self.trajectory_points)):
                displacement = self.calculate_displacement(self.trajectory_points[i-1], self.trajectory_points[i], pixel_to_distance_ratio)
                speed = (displacement / self.time_between_frames) * 3.6
                speeds.append(speed)

            print("Ball Speed:", np.mean(speeds), "km/h")
        else:
            print("Not enough points detected to calculate speed.")

    def create_output_video(self, output_filename='output_video.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frames = sorted(os.listdir(self.processed_frame_dir), key=lambda x: int(re.sub('\\D', '', x)))

        first_frame = cv2.imread(os.path.join(self.processed_frame_dir, frames[0]))
        height, width, _ = first_frame.shape
        size = (width, height)

        out = cv2.VideoWriter(output_filename, fourcc, self.fps, size)
        for frame_filename in frames:
            frame = cv2.imread(os.path.join(self.processed_frame_dir, frame_filename))
            out.write(frame)

        out.release()
        print(f"Video saved as {output_filename}")

if __name__ == "__main__":
    video_path = 'Netpractice.mp4'  # Replace with actual video file path
    model_path = os.path.join('runs', 'detect', 'train8', 'weights', 'best.pt')
    tracker = BallTracking(video_path, model_path)
    tracker.process_video()
    tracker.calculate_speeds()
    tracker.create_output_video()
