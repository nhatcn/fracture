# from object_tracking import ObjectTracking
# from ultralytics import YOLO

# INPUT_PATH = "assets/video/fpt-test.mp4"
# OUTPUT_PATH = "assets/video/fpt-test-ouput-best12adas.mp4"

# if __name__ == "__main__":
#     obj = ObjectTracking(INPUT_PATH, OUTPUT_PATH)
#     obj.process()


from object_tracking import ObjectTracking
from ultralytics import YOLO

INPUT_PATH = "assets/video/fpt-test.mp4"
OUTPUT_PATH = "assets/video/fpt-test-ouput-best12adas.mp4"

if __name__ == "__main__":
    # Chạy với YOLOv8m (best7.pt)
    obj = ObjectTracking(INPUT_PATH, OUTPUT_PATH, model_path="best7.pt")
    obj.process()

    # Chạy với YOLOv8m_ResBlock_CBAM
    obj_cbam = ObjectTracking(INPUT_PATH, "assets/video/fpt-test-output-cbam.mp4", model_path="yolov8m_resblock_cbam.pt")
    obj_cbam.process()

    print("Processing complete. Results saved for comparison.")