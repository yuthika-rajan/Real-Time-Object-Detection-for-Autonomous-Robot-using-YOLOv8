import cv2
import time
import torch
from ultralytics import YOLO

def benchmark(model_name, video_path="sample.mp4", num_frames=200):
    print(f"Benchmarking {model_name}...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return None

    # Warmup
    print("Warming up...")
    for _ in range(20):
        ret, frame = cap.read()
        if not ret: break
        model(frame, verbose=False)

    print(f"Running inference for {num_frames} frames...")
    start_time = time.time()
    
    count = 0
    inference_times = []
    
    while count < num_frames:
        ret, frame = cap.read()
        if not ret: break

        t0 = time.time()
        results = model(frame, verbose=False)
        t1 = time.time()
        
        inference_times.append((t1 - t0) * 1000) # ms
        count += 1

    total_time = time.time() - start_time
    avg_fps = count / total_time
    avg_inference = sum(inference_times) / len(inference_times)
    
    cap.release()
    return avg_fps, avg_inference

def main():
    print("--- YOLOv8 Performance Benchmark ---")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    results = {}
    
    # Benchmark Nano
    fps_n, inf_n = benchmark('yolov8n.pt')
    results['YOLOv8n'] = (fps_n, inf_n)
    
    # Benchmark Small (for comparison)
    fps_s, inf_s = benchmark('yolov8s.pt')
    results['YOLOv8s'] = (fps_s, inf_s)

    print("\n\n=== FINAL RESULTS FOR POSTER ===")
    print("| Model | Average FPS | Inference Time (ms) |")
    print("|-------|-------------|---------------------|")
    for model, (fps, inf) in results.items():
        print(f"| {model} | {fps:.2f} | {inf:.2f} |")
    print("================================")

if __name__ == "__main__":
    main()
