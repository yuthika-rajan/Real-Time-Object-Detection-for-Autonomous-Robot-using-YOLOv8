import cv2
import time
import numpy as np
from ultralytics import YOLO

def main():
    print("Initializing Autonomous Robot Vision System...")
    
    # 1. Load Model
    print("Loading YOLOv8 model...")
    try:
        model = YOLO('yolov8n.pt') 
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Open Webcam (0 is usually the built-in laptop camera)
    print("Opening Webcam (Source 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam. Please ensure it is connected.")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Zones for Robot Logic (Left, Center, Right)
    left_x = width // 3
    right_x = 2 * (width // 3)

    print("System Active. Press 'q' to exit.")
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # 3. Inference
        results = model(frame, stream=True, verbose=False)
        
        # Logic Variables
        command = "IDLE - SEARCHING"
        cmd_color = (200, 200, 200) # Gray
        target_center = None
        max_area = 0

        # Draw Zone Lines (Subtle)
        cv2.line(frame, (left_x, 0), (left_x, height), (100, 100, 100), 1)
        cv2.line(frame, (right_x, 0), (right_x, height), (100, 100, 100), 1)

        # 4. Process Detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                # Filter for Class 0 -> 'person' (Can change to others)
                if cls == 0: 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Calculate Area and Center
                    area = (x2 - x1) * (y2 - y1)
                    center_x = (x1 + x2) // 2

                    # Heuristic: Robot tracks the largest (closest) person
                    if area > max_area:
                        max_area = area
                        target_center = center_x
                        
                        # Visual: Green Box for target
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Target {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        # Visual: Red Box for background people
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # 5. Robot Decision Logic
        if target_center is not None:
            # Check proximity (If object takes up > 30% of screen, STOP)
            screen_area = width * height
            if max_area > screen_area * 0.30:
                command = "STOP - OBSTACLE"
                cmd_color = (0, 0, 255) # Red
            elif target_center < left_x:
                command = "TURN LEFT"
                cmd_color = (0, 255, 255) # Yellow
            elif target_center > right_x:
                command = "TURN RIGHT"
                cmd_color = (0, 255, 255) # Yellow
            else:
                command = "MOVE FORWARD"
                cmd_color = (0, 255, 0) # Green

        # 6. Dashboard Overlay
        # Create a top info bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (30, 30, 30), -1)
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Display Text
        # Command (Center)
        cv2.putText(frame, f"CMD: {command}", (width//2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cmd_color, 3)
        
        # FPS (Left)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Mode (Right)
        cv2.putText(frame, "AUTONOMOUS MODE", (width - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        # Show the frame
        cv2.imshow('YOLOv8 Autonomous Robot Vision', frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
