import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics.nn.modules.conv
import math
import os
import numpy as np
import argparse
import cv2

from bytetrack.byte_track import ByteTrack, STrack
from dataclasses import dataclass

from supervision import (
    ColorPalette, Point, VideoInfo, VideoSink,
    get_video_frames_generator, BoundingBoxAnnotator, LabelAnnotator,
    TraceAnnotator, LineZone, LineZoneAnnotator, Detections
)

from typing import List
from ultralytics import YOLO
from tqdm import tqdm

# ------------------- ECAAttention Definition -------------------
class ECAAttention(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

# Register ECAAttention to Ultralytics
setattr(ultralytics.nn.modules.conv, "ECAAttention", ECAAttention)

# ------------------- ObjectTracking Class -------------------
class ObjectTracking:
    def __init__(self, input_video_path, output_video_path) -> None:
        print(f"Checking input video: {input_video_path}")
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        
        self.model = YOLO("bestCOCO.pt").to("cuda")
        self.model.fuse()
        print("Model loaded successfully")
        
        self.CLASS_NAMES_DICT = self.model.model.names
        self.CLASS_ID = [0]  # Person class 

        self.input_video_path = input_video_path
        self.video_info = VideoInfo.from_video_path(self.input_video_path)
        self.width = self.video_info.width
        self.height = self.video_info.height
        print(f"Video info: {self.width}x{self.height}")
        
        self.LINE_START = Point(0, int(self.height * 0.8))
        self.LINE_END = Point(self.width, int(self.height * 0.8))
        self.LINE_START = Point(int(self.width * 1), int(self.height * 0.9))
        self.LINE_END   = Point(int(self.width * 0.56), int(self.height * 0.12))
        self.output_video_path = output_video_path

        self.byte_tracker = ByteTrack(
            track_activation_threshold=0.2,
            lost_track_buffer=60,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        self.generator = get_video_frames_generator(self.input_video_path)
        self.line_zone = LineZone(start=self.LINE_START, end=self.LINE_END)
        self.box_annotator = BoundingBoxAnnotator(thickness=1)
        self.label_annotator = LabelAnnotator(text_thickness=1, text_scale=0.3)
        self.trace_annotator = TraceAnnotator(thickness=2, trace_length=50)
        self.line_zone_annotator = LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.6)

    def callback(self, frame, index):
        results = self.model(frame, imgsz=1280, verbose=False)[0]
        detections = Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, self.CLASS_ID)]

        detections = self.byte_tracker.update_with_detections(detections)

        if detections.tracker_id is None or len(detections.tracker_id) != len(detections.class_id):
            detections.tracker_id = [0] * len(detections.class_id)

        labels = [
            f"#{tracker_id} {self.model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id in zip(
                detections.confidence, detections.class_id, detections.tracker_id
            )
        ]

        # Annotate
        annotated_frame = self.trace_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        self.line_zone.trigger(detections)
        person_count = self.line_zone.in_count - self.line_zone.out_count

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
#     get_video_frames_generator, BoundingBoxAnnotator, LabelAnnotator,
#     TraceAnnotator, LineZone, LineZoneAnnotator, Detections
# )

# from typing import List
# from ultralytics import YOLO
# from tqdm import tqdm

# # ------------------- ResBlock_CBAM Definition -------------------
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CBAM(nn.Module):
#     def __init__(self, in_planes, ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.ca = ChannelAttention(in_planes, ratio)
#         self.sa = SpatialAttention(kernel_size)

#     def forward(self, x):
#         x = x * self.ca(x)
#         x = x * self.sa(x)
#         return x

# class ResBlock_CBAM(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResBlock_CBAM, self).__init__()
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels // 4),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels // 4),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.cbam = CBAM(out_channels)

#     def forward(self, x):
#         out = self.bottleneck(x)
#         out = self.cbam(out)
#         return out

# # Đăng ký ResBlock_CBAM vào Ultralytics
# setattr(ultralytics.nn.modules.conv, "ResBlock_CBAM", ResBlock_CBAM)

# # ------------------- ObjectTracking Class -------------------
# class ObjectTracking:
#     def __init__(self, input_video_path, output_video_path) -> None:
#         print(f"Checking input video: {input_video_path}")
#         if not os.path.exists(input_video_path):
#             raise FileNotFoundError(f"Input video not found: {input_video_path}")
        
#         self.model = YOLO("best_v8_CBAM.pt").to("cuda")
#         self.model.fuse()
#         print(f"Model loaded successfully")
#         print(f"Class names: {self.model.model.names}")
        
#         self.CLASS_NAMES_DICT = self.model.model.names
#         self.CLASS_ID = [0]  # Person class

#         self.input_video_path = input_video_path
#         self.video_info = VideoInfo.from_video_path(self.input_video_path)
#         self.width = self.video_info.width
#         self.height = self.video_info.height
#         print(f"Video info: {self.width}x{self.height}")
        
#         self.LINE_START = Point(0, int(self.height * 0.7))
#         self.LINE_END = Point(self.width, int(self.height * 0.7))
#         self.output_video_path = output_video_path

#         self.byte_tracker = ByteTrack(
#             track_activation_threshold=0.2,
#             lost_track_buffer=60,
#             minimum_matching_threshold=0.8,
#             frame_rate=30
#         )

#         self.generator = get_video_frames_generator(self.input_video_path)
#         self.line_zone = LineZone(start=self.LINE_START, end=self.LINE_END)
#         self.box_annotator = BoundingBoxAnnotator(thickness=1)
#         self.label_annotator = LabelAnnotator(text_thickness=1, text_scale=0.3)
#         self.trace_annotator = TraceAnnotator(thickness=2, trace_length=50)
#         self.line_zone_annotator = LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.6)

#         self.detection_results = []
#         self.tracking_results = []

#     def callback(self, frame, index):
#         results = self.model(frame, imgsz=1280, verbose=False)[0]
#         print(f"Frame {index} - Raw detections: {results.boxes.xyxy}")
#         print(f"Frame {index} - Class IDs: {results.boxes.cls}")
#         print(f"Frame {index} - Confidences: {results.boxes.conf}")
#         detections = Detections.from_ultralytics(results)
#         print(f"Frame {index} - After Detections.from_ultralytics: {detections.xyxy}")
#         detections = detections[np.isin(detections.class_id, self.CLASS_ID)]
#         print(f"Frame {index} - After class filter: {detections.xyxy}")

#         xywh = []
#         for box in detections.xyxy:
#             x1, y1, x2, y2 = box
#             w = x2 - x1
#             h = y2 - y1
#             x = x1 + w / 2
#             y = y1 + h / 2
#             xywh.append([x, y, w, h])

#         for bbox, conf, class_id in zip(xywh, detections.confidence, detections.class_id):
#             x, y, w, h = bbox
#             self.detection_results.append([index + 1, x - w/2, y - h/2, w, h, conf, class_id])

#         detections = self.byte_tracker.update_with_detections(detections)

#         if detections.tracker_id is None or len(detections.tracker_id) != len(detections.class_id):
#             detections.tracker_id = [0] * len(detections.class_id)

#         tracking_xywh = []
#         for box in detections.xyxy:
#             x1, y1, x2, y2 = box
#             w = x2 - x1
#             h = y2 - y1
#             x = x1 + w / 2
#             y = y1 + h / 2
#             tracking_xywh.append([x, y, w, h])

#         for tracker_id, bbox in zip(detections.tracker_id, tracking_xywh):
#             x, y, w, h = bbox
#             self.tracking_results.append([index + 1, tracker_id, x - w/2, y - h/2, w, h])

#         labels = [
#             f"#{tracker_id} {self.model.model.names[class_id]} {confidence:0.2f}"
#             for confidence, class_id, tracker_id in zip(
#                 detections.confidence, detections.class_id, detections.tracker_id
#             )
#         ]

#         annotated_frame = self.trace_annotator.annotate(scene=frame.copy(), detections=detections)
#         annotated_frame = self.box_annotator.annotate(
#             scene=annotated_frame,
#             detections=detections
#         )
#         annotated_frame = self.label_annotator.annotate(
#             scene=annotated_frame,
#             detections=detections,
#             labels=labels
#         )

#         self.line_zone.trigger(detections)
#         person_count = self.line_zone.in_count - self.line_zone.out_count

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

#         base_name = os.path.splitext(os.path.basename(self.output_video_path))[0]
#         detection_output_file = f"{base_name}_detections.txt"
#         tracking_output_file = f"{base_name}_tracking.txt"
#         counting_output_file = f"{base_name}_counting.txt"

#         with open(detection_output_file, 'w') as f:
#             for det in self.detection_results:
#                 f.write(f"{det[0]},{det[1]},{det[2]},{det[3]},{det[4]},{det[5]},{det[6]}\n")

#         with open(tracking_output_file, 'w') as f:
#             for track in self.tracking_results:
#                 f.write(f"{track[0]},{track[1]},{track[2]},{track[3]},{track[4]},{track[5]}\n")

#         with open(counting_output_file, 'w') as f:
#             f.write(f"in_count: {self.line_zone.in_count}\n")
#             f.write(f"out_count: {self.line_zone.out_count}\n")
            
            
            
            
            
            
            
            
            
            
            
            
            
# class ECAAttention(nn.Module):
#     def __init__(self, channel, k_size=3):
#         """
#         Efficient Channel Attention (ECA) module.

#         Args:
#             channel (int): Number of input channels.
#             k_size (int): Size of the 1D convolution kernel (default: 3).
#         """
#         super(ECAAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         k_size = k_size if k_size % 2 else k_size + 1
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = y.squeeze(-1).transpose(-1, -2)
#         y = self.conv(y)
#         y = y.transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         return x * y

# # ------------------- SpatialAttention Definition -------------------
# class SpatialAttention(nn.Module):
#     def __init__(self, k_size=7):
#         """
#         Spatial Attention module.

#         Args:
#             k_size (int): Size of the convolution kernel (default: 7).
#         """
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling across channels
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling across channels
#         y = torch.cat([avg_out, max_out], dim=1)  # Concatenate along channel dimension
#         y = self.conv(y)
#         y = self.sigmoid(y)
#         return x * y

# # Register modules to Ultralytics
# setattr(ultralytics.nn.modules.conv, "ECAAttention", ECAAttention)            