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
    get_video_frames_generator, BoxAnnotator,
    TraceAnnotator, LineZone, LineZoneAnnotator, Detections
)

from typing import List
from ultralytics import YOLO
from tqdm import tqdm

# ------------------- CBAM Attention Definition -------------------
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
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class CBAMAttention(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAMAttention, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# Register CBAMAttention to Ultralytics
setattr(ultralytics.nn.modules.conv, "CBAMAttention", CBAMAttention)

class ObjectTracking:
    def __init__(self, input_video_path, output_video_path) -> None:
        print(f"Checking input video: {input_video_path}")
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        # Load model đã train với CBAM
        self.model = YOLO("best12epoch.pt").to("cuda")
        self.model.fuse()
        print("Model loaded successfully")

        self.CLASS_NAMES_DICT = self.model.model.names
        self.CLASS_ID = [0]  

        self.input_video_path = input_video_path
        self.video_info = VideoInfo.from_video_path(self.input_video_path)
        self.width = self.video_info.width
        self.height = self.video_info.height
        print(f"Video info: {self.width}x{self.height}")

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
        self.box_annotator = BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.3)
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

        annotated_frame = self.trace_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

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
