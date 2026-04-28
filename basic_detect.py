import cv2
from ultralytics import YOLO

def main():
    print("Initializing Basic Real-Time Object Detection...")
    
    # Load the model
    # yolov8n.pt is the fastest model, ensuring real-time performance
    model = YOLO('yolov8n.pt') 

    # Start detection on source 0 (Webcam)
    # show=True lets YOLO handle the visualization logic (bounding boxes, labels) efficiently
    # stream=True ensures it processes frames as a generator for performance
    print("Starting Webcam Feed. Press 'q' to exit.")
    
    # We use the built-in 'track' or 'predict' method which handles everything
    results = model.predict(source="0", show=True, stream=True)
    
    # Keep the script running to keep the window open
    for r in results:
        pass # The 'show=True' above handles the display updates automatically

if __name__ == "__main__":
    main()
