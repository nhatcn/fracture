from object_tracking import ObjectTracking
from ultralytics import YOLO

INPUT_PATH = "assets/video/fpt-test.mp4"
OUTPUT_PATH = "assets/video/fpt-test-ouput-best12adas.mp4"

if __name__ == "__main__":
    obj = ObjectTracking(INPUT_PATH, OUTPUT_PATH)
    obj.process() t