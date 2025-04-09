import os
import re
import numpy as np
from ultralytics import YOLO
from object_tracking import ObjectTracking
from supervision import Detections, get_video_frames_generator, VideoSink
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict

INPUT_PATH = "assets/video/fpt-test.mp4"
OUTPUT_PATH_BEST5 = "assets/video/6415701305402-output-best5.mp4"
OUTPUT_PATH_YOLOV8M = "assets/video/6415701305402-output-yolov8m.mp4"
LABELS_FOLDER = "train/labels"  # Replace with the actual path if needed
MAX_FRAMES = 5500  # Increased to include frames from 2736 to 5464

def load_ground_truth_labels(labels_folder, max_frames):
    ground_truth = {}
    labeled_frames = []
    print(f"Checking labels in: {labels_folder}")
    
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    print(f"Found {len(label_files)} label files: {label_files[:5]}")
    
    label_files_with_frame = []
    for label_file in label_files:
        match = re.search(r'-(\d+)_jpg\.rf\.', label_file)
        if match:
            frame_num = int(match.group(1)) - 1
            if frame_num < max_frames:
                label_files_with_frame.append((frame_num, label_file))
    
    label_files_with_frame.sort(key=lambda x: x[0])
    for frame_idx, label_file in label_files_with_frame:
        label_path = os.path.join(labels_folder, label_file)
        with open(label_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
            boxes = [[float(x) for x in label[1:]] for label in labels]  # [center_x, center_y, width, height]
            class_ids = [int(label[0]) for label in labels]
            ground_truth[frame_idx] = {"boxes": boxes, "class_ids": class_ids}
            labeled_frames.append(frame_idx)
    
    print(f"Number of frames with labels: {len(labeled_frames)}")
    return ground_truth, labeled_frames

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def xyxy_to_center_wh(box_xyxy, img_width, img_height):
    x_min, y_min, x_max, y_max = box_xyxy
    center_x = (x_min + x_max) / 2 / img_width
    center_y = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [center_x, center_y, width, height]

def calculate_mae_mse(pred_boxes, gt_boxes, img_width, img_height):
    mae_list = []
    mse_list = []
    for pred_box, gt_box in zip(pred_boxes, gt_boxes):
        pred_box_norm = xyxy_to_center_wh(pred_box, img_width, img_height)
        errors = [abs(p - g) for p, g in zip(pred_box_norm, gt_box)]
        mae = np.mean(errors)
        mse = np.mean([e ** 2 for e in errors])
        mae_list.append(mae)
        mse_list.append(mse)
        print(f"Predicted box: {pred_box_norm}, Ground truth box: {gt_box}, Errors: {errors}, MAE: {mae}, MSE: {mse}")
    mae_result = np.mean(mae_list) if mae_list else 0
    mse_result = np.mean(mse_list) if mse_list else 0
    print(f"MAE list: {mae_list[:5]}... (total {len(mae_list)})")
    print(f"MSE list: {mse_list[:5]}... (total {len(mse_list)})")
    return mae_result, mse_result

def calculate_map(predictions, ground_truth, labeled_frames, img_width, img_height, iou_thresholds=[0.5, 0.75]):
    ap_dict = defaultdict(list)
    for frame_idx in labeled_frames:
        pred = predictions.get(frame_idx, {"boxes": [], "class_ids": [], "confidences": []})
        gt = ground_truth.get(frame_idx, {"boxes": [], "class_ids": []})
        pred_boxes = pred.get("boxes", [])
        pred_class_ids = pred.get("class_ids", [])
        pred_confidences = pred.get("confidences", [1.0] * len(pred_boxes))
        gt_boxes = gt["boxes"]
        gt_class_ids = gt["class_ids"]
        for iou_th in iou_thresholds:
            matched = []
            for i, (p_box, p_class, p_conf) in enumerate(zip(pred_boxes, pred_class_ids, pred_confidences)):
                max_iou = 0
                match_idx = -1
                for j, (g_box, g_class) in enumerate(zip(gt_boxes, gt_class_ids)):
                    if p_class == g_class:
                        iou = calculate_iou(xyxy_to_center_wh(p_box, img_width, img_height), g_box)
                        if iou > max_iou and iou >= iou_th:
                            max_iou = iou
                            match_idx = j
                if match_idx != -1:
                    matched.append((p_conf, 1))
                else:
                    matched.append((p_conf, 0))
            for j in range(len(gt_boxes)):
                if j not in [m[1] for m in matched if m[1] != -1]:
                    matched.append((0, 0))
            matched = sorted(matched, key=lambda x: x[0], reverse=True)
            precisions = []
            recalls = []
            tp = 0
            fp = 0
            total_gt = len(gt_boxes)
            for conf, label in matched:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
                precisions.append(tp / (tp + fp) if tp + fp > 0 else 0)
                recalls.append(tp / total_gt if total_gt > 0 else 0)
            ap = np.trapz(precisions, recalls) if precisions and recalls else 0
            ap_dict[iou_th].append(ap)
    return {iou_th: np.mean(aps) for iou_th, aps in ap_dict.items()}

def evaluate_model(predictions, ground_truth, labeled_frames, img_width, img_height):
    y_true = []
    y_pred = []
    mae_boxes = []
    mse_boxes = []
    total_detections = 0
    total_ground_truth = 0

    for frame_idx in labeled_frames:
        pred = predictions.get(frame_idx, {"boxes": [], "class_ids": []})
        gt = ground_truth.get(frame_idx, {"boxes": [], "class_ids": []})
        pred_boxes = pred.get("boxes", [])
        pred_class_ids = pred.get("class_ids", [])
        gt_boxes = gt["boxes"]
        gt_class_ids = gt["class_ids"]

        total_detections += len(pred_boxes)
        total_ground_truth += len(gt_boxes)

        matched = []
        for i, (p_box, p_class) in enumerate(zip(pred_boxes, pred_class_ids)):
            max_iou = 0
            match_idx = -1
            for j, (g_box, g_class) in enumerate(zip(gt_boxes, gt_class_ids)):
                if p_class == g_class:
                    iou = calculate_iou(xyxy_to_center_wh(p_box, img_width, img_height), g_box)
                    if iou > max_iou and iou >= 0.5:
                        max_iou = iou
                        match_idx = j
            if match_idx != -1:
                y_true.append(1)
                y_pred.append(1)
                matched.append(match_idx)
                mae, mse = calculate_mae_mse([p_box], [gt_boxes[match_idx]], img_width, img_height)
                mae_boxes.append(mae)
                mse_boxes.append(mse)
            else:
                y_true.append(0)
                y_pred.append(1)
        for j in range(len(gt_boxes)):
            if j not in matched:
                y_true.append(1)
                y_pred.append(0)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    mae = np.mean(mae_boxes) if mae_boxes else 0
    mse = np.mean(mse_boxes) if mse_boxes else 0
    mAP = calculate_map(predictions, ground_truth, labeled_frames, img_width, img_height)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae,
        "mse": mse,
        "mAP": mAP,
        "total_detections": total_detections,
        "total_ground_truth": total_ground_truth
    }

if __name__ == "__main__":
    # Initialize the two models
    obj_best5 = ObjectTracking(INPUT_PATH, OUTPUT_PATH_BEST5)
    obj_best5.model = YOLO("best5.pt").to("cuda")
    obj_yolov8m = ObjectTracking(INPUT_PATH, OUTPUT_PATH_YOLOV8M)
    obj_yolov8m.model = YOLO("best_of_v8.pt").to("cuda")

    print("Best5 class names:", obj_best5.model.model.names)
    print("YOLOv8m class names:", obj_yolov8m.model.model.names)

    # Both models use only class 0
    obj_best5.CLASS_ID = [0]  # Class 0: "person"
    obj_yolov8m.CLASS_ID = [0]  # Class 0: "person"

    video_info = obj_best5.video_info
    img_width, img_height = video_info.width, video_info.height
    print(f"Video resolution: {img_width}x{img_height}")

    ground_truth, labeled_frames = load_ground_truth_labels(LABELS_FOLDER, MAX_FRAMES)
    if not labeled_frames:
        print("No labeled frames found. Exiting...")
        exit()

    predictions_best5 = {}
    predictions_yolov8m = {}
    generator = get_video_frames_generator(INPUT_PATH)

    for frame_idx, frame in enumerate(generator):
        if frame_idx >= MAX_FRAMES:
            break
        if frame_idx in labeled_frames:
            # Best5 predictions
            results_best5 = obj_best5.model(frame, imgsz=1280, verbose=False)[0]
            det_best5 = Detections.from_ultralytics(results_best5)
            det_best5 = det_best5[np.isin(det_best5.class_id, obj_best5.CLASS_ID)]
            print(f"Frame {frame_idx} - Best5 detections (class 0 - pedestrian): {len(det_best5.xyxy)}")
            predictions_best5[frame_idx] = {
                "boxes": det_best5.xyxy.tolist(),
                "class_ids": det_best5.class_id.tolist(),
                "confidences": det_best5.confidence.tolist()
            }
            # YOLOv8m predictions
            results_yolov8m = obj_yolov8m.model(frame, imgsz=1280, verbose=False)[0]
            det_yolov8m = Detections.from_ultralytics(results_yolov8m)
            det_yolov8m = det_yolov8m[np.isin(det_yolov8m.class_id, obj_yolov8m.CLASS_ID)]
            print(f"Frame {frame_idx} - YOLOv8m detections (class 0 - person): {len(det_yolov8m.xyxy)}")
            predictions_yolov8m[frame_idx] = {
                "boxes": det_yolov8m.xyxy.tolist(),
                "class_ids": det_yolov8m.class_id.tolist(),
                "confidences": det_yolov8m.confidence.tolist()
            }

    # Evaluate models
    metrics_best5 = evaluate_model(predictions_best5, ground_truth, labeled_frames, img_width, img_height)
    metrics_yolov8m = evaluate_model(predictions_yolov8m, ground_truth, labeled_frames, img_width, img_height)

    # Print performance comparison
    print(f"\nPerformance comparison ({len(labeled_frames)} labeled frames):")
    print(f"Best5 (Pedestrian) - Total detections: {metrics_best5['total_detections']}, Total ground truth: {metrics_best5['total_ground_truth']}")
    print(f"Best5 - Precision: {metrics_best5['precision']:.4f}, Recall: {metrics_best5['recall']:.4f}, F1: {metrics_best5['f1']:.4f}")
    print(f"      - MAE: {metrics_best5['mae']:.6f}, MSE: {metrics_best5['mse']:.6f}")
    print(f"      - mAP: {metrics_best5['mAP']}")
    print(f"YOLOv8m (Person) - Total detections: {metrics_yolov8m['total_detections']}, Total ground truth: {metrics_yolov8m['total_ground_truth']}")
    print(f"YOLOv8m - Precision: {metrics_yolov8m['precision']:.4f}, Recall: {metrics_yolov8m['recall']:.4f}, F1: {metrics_yolov8m['f1']:.4f}")
    print(f"        - MAE: {metrics_yolov8m['mae']:.6f}, MSE: {metrics_yolov8m['mse']:.6f}")
    print(f"        - mAP: {metrics_yolov8m['mAP']}")

    # Process video output
    print("Processing video for Best5...")
    with VideoSink(target_path=OUTPUT_PATH_BEST5, video_info=video_info) as sink:
        generator = get_video_frames_generator(INPUT_PATH)
        for idx, frame in enumerate(generator):
            if idx >= MAX_FRAMES:
                break
            result_frame = obj_best5.callback(frame, idx)
            sink.write_frame(frame=result_frame)

    print("Processing video for YOLOv8m...")
    with VideoSink(target_path=OUTPUT_PATH_YOLOV8M, video_info=video_info) as sink:
        generator = get_video_frames_generator(INPUT_PATH)
        for idx, frame in enumerate(generator):
            if idx >= MAX_FRAMES:
                break
            result_frame = obj_yolov8m.callback(frame, idx)
            sink.write_frame(frame=result_frame)