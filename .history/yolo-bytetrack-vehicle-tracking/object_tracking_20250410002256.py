# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import ultralytics.nn.modules.conv
# import math
# import os
# import numpy as np
# import argparse
# import cv2

# from bytetrack.byte_track import ByteTrack, STrack
# from dataclasses import dataclass

# from supervision import (
#     ColorPalette, Point, VideoInfo, VideoSink,
#     get_video_frames_generator, BoxAnnotator,
#     TraceAnnotator, LineZone, LineZoneAnnotator, Detections
# )

# from typing import List
# from ultralytics import YOLO
# from tqdm import tqdm

# # ------------------- ECAAttention Definition -------------------
# class ECAAttention(nn.Module):
#     def __init__(self, channel, gamma=2, b=1):
#         super(ECAAttention, self).__init__()
#         kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = y.squeeze(-1).transpose(-1, -2)
#         y = self.conv(y)
#         y = y.transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         return x * y

# # Register ECAAttention to Ultralytics
# setattr(ultralytics.nn.modules.conv, "ECAAttention", ECAAttention)

# # ------------------- ObjectTracking Class -------------------
# class ObjectTracking:
#     def __init__(self, input_video_path, output_video_path) -> None:
#         print(f"Checking input video: {input_video_path}")
#         if not os.path.exists(input_video_path):
#             raise FileNotFoundError(f"Input video not found: {input_video_path}")
        
#         self.model = YOLO("best7.pt").to("cuda")
#         self.model.fuse()
#         print("Model loaded successfully")
        
#         self.CLASS_NAMES_DICT = self.model.model.names
#         self.CLASS_ID = [0]  # Person  class 

#         self.input_video_path = input_video_path
#         self.video_info = VideoInfo.from_video_path(self.input_video_path)
#         self.width = self.video_info.width
#         self.height = self.video_info.height
#         print(f"Video info: {self.width}x{self.height}")
        
#         self.LINE_START = Point(int(self.width * 1), int(self.height * 0.9))
#         self.LINE_END   = Point(int(self.width * 0.56), int(self.height * 0.12))
        
#         # self.LINE_START = Point( 0, int(self.height * 0.7))
#         # self.LINE_END = Point(self.width ,int(self.height * 0.7))
#         self.output_video_path = output_video_path

#         # Updated ByteTrack init
#         self.byte_tracker = ByteTrack(
#             track_activation_threshold=0.2,
#             lost_track_buffer=60,
#             minimum_matching_threshold=0.8,
#             frame_rate=30
#         )

#         self.generator = get_video_frames_generator(self.input_video_path)
#         self.line_zone = LineZone(start=self.LINE_START, end=self.LINE_END)
#         self.box_annotator = BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.3)
#         self.trace_annotator = TraceAnnotator(thickness=2, trace_length=50)
#         self.line_zone_annotator = LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.6)

#     def callback(self, frame, index):
#         results = self.model(frame, imgsz=2560, verbose=False)[0]
#         detections = Detections.from_ultralytics(results)
#         detections = detections[np.isin(detections.class_id, self.CLASS_ID)]

#         # ByteTrack tracking
#         detections = self.byte_tracker.update_with_detections(detections)

#         # Ensure detections.tracker_id is not None and is same length
#         if detections.tracker_id is None or len(detections.tracker_id) != len(detections.class_id):
#             detections.tracker_id = [0] * len(detections.class_id)

#         # Create labels safely
#         labels = [
#             f"#{tracker_id} {self.model.model.names[class_id]} {confidence:0.2f}"
#             for confidence, class_id, tracker_id in zip(
#                 detections.confidence, detections.class_id, detections.tracker_id
#             )
#         ]

#         # Annotate
#         annotated_frame = self.trace_annotator.annotate(scene=frame.copy(), detections=detections)
#         annotated_frame = self.box_annotator.annotate(
#             scene=annotated_frame,
#             detections=detections,
#             labels=labels
#         )

#         self.line_zone.trigger(detections)
#         person_count = self.line_zone.in_count - self.line_zone.out_count

#         # Display counts
#         cv2.putText(
#             annotated_frame,
#             f"People Count: {person_count}",
#             (10, 50),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 255),
#             2
#         )
#         cv2.putText(
#             annotated_frame,
#             f"Down: {self.line_zone.in_count} | Up: {self.line_zone.out_count}",
#             (10, 100),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 0),
#             2
#         )

#         return self.line_zone_annotator.annotate(annotated_frame, line_counter=self.line_zone)

#     def process(self):
#         print(f"Opening video sink: {self.output_video_path}")
#         with VideoSink(target_path=self.output_video_path, video_info=self.video_info) as sink:
#             for index, frame in enumerate(get_video_frames_generator(source_path=self.input_video_path)):
#                 print(f"Processing frame {index}")
#                 try:
#                     result_frame = self.callback(frame, index)
#                     sink.write_frame(frame=result_frame)
#                 except Exception as e:
#                     print(f"Error at frame {index}: {e}")
#                     continue
#         print("Video processing finished")




import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics.nn.modules.conv
import math
import os
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from bytetrack.byte_track import ByteTrack, STrack
from dataclasses import dataclass
from supervision import (
    ColorPalette, Point, VideoInfo, VideoSink,
    get_video_frames_generator, BoxAnnotator,
    TraceAnnotator, LineZone, LineZoneAnnotator, Detections
)
from typing import List
from ultralytics import YOLO
from tqdm import tqdm

# ------------------- Hàm hỗ trợ -------------------
def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def assign_ids_to_objects(labels_folder, video_path, output_gt_file, expected_frames=1050):
    width, height = get_video_dimensions(video_path)
    label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith('.txt')])
    if len(label_files) != expected_frames:
        raise ValueError(f"Expected {expected_frames} label files, but found {len(label_files)}")

    gt_data = []
    current_id = 1
    prev_boxes = []
    prev_ids = []

    for frame_idx, label_file in enumerate(label_files, 1):
        label_path = os.path.join(labels_folder, label_file)
        boxes = []

        with open(label_path, 'r') as f:
            for line in f:
                class_id, center_x, center_y, w, h = map(float, line.strip().split())
                if int(class_id) != 0:  # Chỉ lấy class "person"
                    continue
                center_x *= width
                center_y *= height
                w *= width
                h *= height
                x1 = center_x - w / 2
                y1 = center_y - h / 2
                boxes.append([x1, y1, x1 + w, y1 + h])

        if not prev_boxes:
            for box in boxes:
                gt_data.append([frame_idx, current_id, box[0], box[1], box[2] - box[0], box[3] - box[1]])
                prev_ids.append(current_id)
                current_id += 1
        else:
            iou_matrix = np.zeros((len(boxes), len(prev_boxes)))
            for i, box in enumerate(boxes):
                for j, prev_box in enumerate(prev_boxes):
                    iou_matrix[i, j] = iou(box, prev_box)

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched = set()
            new_ids = []

            for i, j in zip(row_ind, col_ind):
                if iou_matrix[i, j] > 0.5:
                    gt_data.append([frame_idx, prev_ids[j], boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]])
                    matched.add(i)
                    new_ids.append(prev_ids[j])
                else:
                    new_ids.append(None)

            for i in range(len(boxes)):
                if i not in matched:
                    gt_data.append([frame_idx, current_id, boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]])
                    new_ids.append(current_id)
                    current_id += 1

            prev_ids = new_ids

        prev_boxes = boxes

    with open(output_gt_file, 'w') as f:
        for frame, obj_id, x, y, w, h in gt_data:
            f.write(f"{frame},{obj_id},{x},{y},{w},{h},1,-1,-1,-1\n")
    print(f"Ground truth saved to: {output_gt_file}")

def compute_counting_ground_truth(gt_file, line_start_x, line_end_x, height):
    gt_data = []
    with open(gt_file, 'r') as f:
        for line in f:
            frame, obj_id, x, y, w, h, conf, _, _, _ = map(float, line.strip().split(','))
            gt_data.append([int(frame), int(obj_id), x + w/2, y + h/2])

    gt_data.sort(key=lambda x: (x[0], x[1]))

    line_y = height * 0.5
    in_count = 0
    out_count = 0
    tracked_ids = set()

    for i in range(len(gt_data) - 1):
        curr_frame, curr_id, curr_x, curr_y = gt_data[i]
        next_frame, next_id, next_x, next_y = gt_data[i + 1]

        if curr_id != next_id or next_frame != curr_frame + 1:
            continue

        if curr_id not in tracked_ids:
            if (curr_y < line_y and next_y >= line_y) and (line_start_x <= curr_x <= line_end_x):
                in_count += 1
                tracked_ids.add(curr_id)
            elif (curr_y >= line_y and next_y < line_y) and (line_start_x <= curr_x <= line_end_x):
                out_count += 1
                tracked_ids.add(curr_id)

    return in_count, out_count

# ------------------- ResBlock_CBAM Definition -------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ResBlock_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_CBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Đăng ký ResBlock_CBAM vào Ultralytics
setattr(ultralytics.nn.modules.conv, "ResBlock_CBAM", ResBlock_CBAM)

# ------------------- ObjectTracking Class -------------------
class ObjectTracking:
    def __init__(self, input_video_path, output_video_path, model_path="best7.pt") -> None:
        print(f"Checking input video: {input_video_path}")
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        
        # Tải mô hình
        self.model = YOLO(model_path).to("cuda")
        self.model.fuse()
        print(f"Model {model_path} loaded successfully")
        
        self.CLASS_NAMES_DICT = self.model.model.names
        self.CLASS_ID = [0]  # Person class 

        self.input_video_path = input_video_path
        self.video_info = VideoInfo.from_video_path(self.input_video_path)
        self.width = self.video_info.width
        self.height = self.video_info.height
        print(f"Video info: {self.width}x{self.height}")
        
        self.LINE_START = Point(int(self.width * 1), int(self.height * 0.9))
        self.LINE_END   = Point(int(self.width * 0.56), int(self.height * 0.12))
        
        self.output_video_path = output_video_path

        # Định nghĩa đường dẫn để lưu kết quả detection, tracking và counting
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        self.detection_output_file = f"{base_name}_detections.txt"
        self.tracking_output_file = f"{base_name}_tracking.txt"
        self.counting_output_file = f"{base_name}_counting.txt"

        # Khởi tạo ByteTrack
        self.byte_tracker = ByteTrack(
            track_activation_threshold=0.2,
            lost_track_buffer=60,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        self.generator = get_video_frames_generator(self.input_video_path)
        self.line_zone = LineZone(start=self.LINE_START, end=self.LINE_END)
        self.box_annotator = BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.3)
        self.trace_annotator = TraceAnnotator(thickness=2, trace_length=50)
        self.line_zone_annotator = LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.6)

        # Lưu trữ kết quả detection và tracking
        self.detection_results = []  # Lưu kết quả detection: [frame, x, y, w, h, conf, class_id]
        self.tracking_results = []   # Lưu kết quả tracking: [frame, id, x, y, w, h]

    def callback(self, frame, index):
        # Dự đoán với mô hình
        results = self.model(frame, imgsz=1280, verbose=False)[0]
        detections = Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, self.CLASS_ID)]

        # Lưu kết quả detection
        for bbox, conf, class_id in zip(detections.xywh, detections.confidence, detections.class_id):
            x, y, w, h = bbox
            self.detection_results.append([index + 1, x - w/2, y - h/2, w, h, conf, class_id])

        # ByteTrack tracking
        detections = self.byte_tracker.update_with_detections(detections)

        # Đảm bảo tracker_id không bị None
        if detections.tracker_id is None or len(detections.tracker_id) != len(detections.class_id):
            detections.tracker_id = [0] * len(detections.class_id)

        # Lưu kết quả tracking
        for tracker_id, bbox in zip(detections.tracker_id, detections.xywh):
            x, y, w, h = bbox
            self.tracking_results.append([index + 1, tracker_id, x - w/2, y - h/2, w, h])

        # Tạo nhãn
        labels = [
            f"#{tracker_id} {self.model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id in zip(
                detections.confidence, detections.class_id, detections.tracker_id
            )
        ]

        # Gắn nhãn và vẽ
        annotated_frame = self.trace_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        # Đếm người
        self.line_zone.trigger(detections)
        person_count = self.line_zone.in_count - self.line_zone.out_count

        # Hiển thị số lượng người
        cv2.putText(
            annotated_frame,
            f"People Count: {person_count}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        cv2.putText(
            annotated_frame,
            f"Down: {self.line_zone.in_count} | Up: {self.line_zone.out_count}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        return self.line_zone_annotator.annotate(annotated_frame, line_counter=self.line_zone)

    def process(self):
        print(f"Opening video sink: {self.output_video_path}")
        with VideoSink(target_path=self.output_video_path, video_info=self.video_info) as sink:
            for index, frame in enumerate(get_video_frames_generator(source_path=self.input_video_path)):
                print(f"Processing frame {index}")
                try:
                    result_frame = self.callback(frame, index)
                    sink.write_frame(frame=result_frame)
                except Exception as e:
                    print(f"Error at frame {index}: {e}")
                    continue
        print("Video processing finished")

        # Lưu kết quả detection vào file
        with open(self.detection_output_file, 'w') as f:
            for det in self.detection_results:
                f.write(f"{det[0]},{det[1]},{det[2]},{det[3]},{det[4]},{det[5]},{det[6]}\n")

        # Lưu kết quả tracking vào file
        with open(self.tracking_output_file, 'w') as f:
            for track in self.tracking_results:
                f.write(f"{track[0]},{track[1]},{track[2]},{track[3]},{track[4]},{track[5]}\n")

        # Lưu kết quả counting vào file
        with open(self.counting_output_file, 'w') as f:
            f.write(f"in_count: {self.line_zone.in_count}\n")
            f.write(f"out_count: {self.line_zone.out_count}\n")