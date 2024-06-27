import torch
import cv2
import time
from picamera import PiCamera
from picamera.array import PiRGBArray

# Load YOLOv5 model (using YOLOv5-Nano)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True, trust_repo=True)

# Function to process and visualize detections
def process_frame(frame):
    results = model(frame)
    labels, coords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    object_count = 0
    for label, coord in zip(labels, coords):
        x1, y1, x2, y2, conf = coord
        x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
        if conf > 0.5:  # Confidence threshold
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.names[int(label)]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            object_count += 1
    cv2.putText(frame, f"Object Count: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, object_count

# Initialize PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(2)  # Allow the camera to warm up

object_count_total = 0
frame_count = 0

# Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    frame_count += 1
    
    # Process frame every 5 seconds
    if frame_count % 150 == 0:  # 30 frames per second * 5 seconds = 150 frames
        image, object_count = process_frame(image)
        object_count_total += object_count  # Accumulate total object count
    
    cv2.imshow('YOLOv5-Nano Detection', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break
    
    rawCapture.truncate(0)

cv2.destroyAllWindows()

# Display total object count after stopping the webcam feed
print(f"Total Objects Detected: {object_count_total}")
