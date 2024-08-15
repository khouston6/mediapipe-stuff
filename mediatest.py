# apt update
# apt install wget python3-dev python3-pip vim lffmpeg libsm6 libxext6 libcudnn8=8.6.*
# pip install mediapipe opencv-python-headless
# pip install --extra-index-url https://pypi.nvidia.com tensorflow[and-cuda]==2.12.0
# wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
# tar -xzvf Tensor*
# mv Tensor* /usr/local/TensorRT-8.6.1
# vim ~/.bashrc
# export PATH=/usr/local/cuda-11.8/bin:/usr/local/TensorRT-8.6.1/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/TensorRT-8.6.1/lib:$LD_LIBRARY_PATH
# source ~/.bashrc

import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Input and output video files
input_video_path = 'input.mp4'
output_video_path = 'output.mp4'

# Open the input video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find pose landmarks
    results = pose.process(image_rgb)

    # Draw the pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Write the frame to the output video file
    out.write(frame)

# Release resources
cap.release()
out.release()

