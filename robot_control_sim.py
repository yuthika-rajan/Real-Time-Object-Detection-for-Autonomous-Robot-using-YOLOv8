import cv2
import time
import numpy as np
from ultralytics import YOLO

def main():
    video_path = "sample.mp4"
    output_path = "robot_simulation_output.avi"
    
    print(f"Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt') 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}.")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    # Zones
    left_x = width // 3
    right_x = 2 * (width // 3)

    print("Starting simulation loop...")
    frame_count = 0
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame, stream=True)
        
        # Logic vars
        command = "IDLE - SEARCHING"
        cmd_color = (200, 200, 200) # Gray
        target_center = None
        max_conf = 0
        max_area = 0

        # Draw Zone Lines (Subtle)
        cv2.line(frame, (left_x, 0), (left_x, height), (100, 100, 100), 1)
        cv2.line(frame, (right_x, 0), (right_x, height), (100, 100, 100), 1)

        # Process detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                # Track 'person' (0). You can add others if needed.
                if cls == 0: 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    area = (x2 - x1) * (y2 - y1)
                    center_x = (x1 + x2) // 2

                    # Heuristic: Follow the largest/closest person
                    if area > max_area:
                        max_area = area
                        target_center = center_x
                        max_conf = conf
                        
                        # Draw Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Target {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Robot Decision Logic
        if target_center is not None:
            # Check proximity (Area based)
            screen_area = width * height
            if max_area > screen_area * 0.25:
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

        # Dashboard Overlay
        # Create a top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (20, 20, 20), -1)
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # FPS Calc
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Draw Dashboard Text
        # 1. System Status / Command (Center)
        cv2.putText(frame, f"CMD: {command}", (width//2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cmd_color, 3)
        
        # 2. FPS (Left)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        
        # 3. Mode (Right)
        cv2.putText(frame, "MODE: AUTONOMOUS", (width - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)

        out.write(frame)
        
        # Save preview
        if frame_count == 120:
             cv2.imwrite("robot_sim_preview.jpg", frame)

        frame_count += 1
        # Shorter run for demo
        if frame_count > 400: 
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Simulation saved to {output_path}")

if __name__ == "__main__":
    main()
