# from object_tracking import ObjectTracking
# from ultralytics import YOLO

# INPUT_PATH = "assets/video/fpt-test.mp4"
# OUTPUT_PATH = "assets/video/fpt-test-ouput-best12adas.mp4"

# if __name__ == "__main__":
#     obj = ObjectTracking(INPUT_PATH, OUTPUT_PATH)
#     obj.process()

from object_tracking import ObjectTracking

INPUT_PATH = "assets/video/fpt-test.mp4"
OUTPUT_PATH = "assets/video/fpt-test-ouput-best12moi.mp4"

if __name__ == "__main__":
    # Chạy với YOLOv8m (best7.pt)
    obj = ObjectTracking(INPUT_PATH, OUTPUT_PATH, model_path="best_of_v8.pt")
    obj.process()

   

    print("Processing complete. Results saved for comparison.")