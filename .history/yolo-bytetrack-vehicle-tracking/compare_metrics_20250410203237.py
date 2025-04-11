import os
import numpy as np
from ultralytics import YOLO
from object_tracking import ObjectTracking
from supervision import Detections, BoundingBoxAnnotator, LabelAnnotator, Point, LineZone
import cv2
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_score, recall_score

def calculate_iou(box1, box2):
    """Tính IoU giữa hai bounding box [x_min, y_min, x_max, y_max]"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def load_ground_truth(label_folder, line_start, line_end):
    """Load ground truth và tính gt_in_count, gt_out_count"""
    gt_dict = {}
    tracks = {}  # Lưu trữ các track: {track_id: [(frame_idx, box, center_y)]}
    next_track_id = 0
    gt_in_count = 0
    gt_out_count = 0

    # Định nghĩa line zone từ line_start và line_end
    line_y = line_start.y  # Giả sử line là đường ngang (y không đổi)

    # Load tất cả frame vào gt_dict trước
    for label_file in sorted(os.listdir(label_folder)):
        if label_file.startswith('frame_') and label_file.endswith('.txt'):
            try:
                frame_part = label_file.split('.')[0]
                frame_str = frame_part.split('_')[1]
                frame_idx = int(frame_str)
            except (IndexError, ValueError) as e:
                print(f"Skipping invalid file name: {label_file}, error: {e}")
                continue
            
            gt_boxes = []
            with open(os.path.join(label_folder, label_file), 'r') as f:
                for line in f:
                    class_id, cx, cy, w, h = map(float, line.strip().split())
                    if class_id == 0:
                        x_min = (cx - w/2) * 960
                        y_min = (cy - h/2) * 540
                        x_max = (cx + w/2) * 960
                        y_max = (cy + h/2) * 540
                        gt_boxes.append([x_min, y_min, x_max, y_max])
            gt_dict[frame_idx] = gt_boxes

    # Theo dõi đối tượng qua các frame
    for frame_idx in sorted(gt_dict.keys()):
        current_boxes = gt_dict[frame_idx]
        current_centers = [(box[0] + box[2]) / 2 for box in current_boxes]  # x trung tâm
        current_centers_y = [(box[1] + box[3]) / 2 for box in current_boxes]  # y trung tâm

        # Khớp các box với các track hiện có
        unmatched_boxes = list(range(len(current_boxes)))
        new_tracks = {}

        for track_id, track in tracks.items():
            last_frame, last_box, last_center_y = track[-1]
            if frame_idx - last_frame > 1:  # Track đã mất quá lâu
                continue

            last_center_x = (last_box[0] + last_box[2]) / 2
            best_iou = 0
            best_idx = -1

            for idx in unmatched_boxes:
                iou = calculate_iou(last_box, current_boxes[idx])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou > 0.5:  # IoU threshold để khớp
                unmatched_boxes.remove(best_idx)
                new_tracks[track_id] = track + [(frame_idx, current_boxes[best_idx], current_centers_y[best_idx])]
                # Kiểm tra vượt qua line
                if len(track) >= 1:
                    prev_y = last_center_y
                    curr_y = current_centers_y[best_idx]
                    if prev_y < line_y and curr_y >= line_y:  # Đi xuống (in)
                        gt_in_count += 1
                    elif prev_y >= line_y and curr_y < line_y:  # Đi lên (out)
                        gt_out_count += 1

        # Tạo track mới cho các box chưa khớp
        for idx in unmatched_boxes:
            new_tracks[next_track_id] = [(frame_idx, current_boxes[idx], current_centers_y[idx])]
            next_track_id += 1

        tracks = new_tracks

    return gt_dict, gt_in_count, gt_out_count

def calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Tính Average Precision (AP) tại IoU threshold"""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    
    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    
    matched_gt = set()
    tp = []
    fp = []
    
    for pred_box in pred_boxes:
        max_iou = 0
        matched_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box[:4], gt_box)
            if iou > max_iou and gt_idx not in matched_gt:
                max_iou = iou
                matched_idx = gt_idx
        
        if max_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched_gt.add(matched_idx)
        else:
            tp.append(0)
            fp.append(1)
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros(len(tp))
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap

def evaluate_model(model_path, input_video, label_folder, output_video):
    """Đánh giá một model"""
    tracker = ObjectTracking(input_video, output_video)
    tracker.model = YOLO(model_path).to("cuda")
    tracker.model.fuse()
    
    # Load ground truth và tính gt_in_count, gt_out_count
    gt_dict, gt_in_count, gt_out_count = load_ground_truth(label_folder, tracker.LINE_START, tracker.LINE_END)
    
    all_preds = []
    all_gts = []
    tp_count, fp_count, fn_count = 0, 0, 0
    total_gt_count = 0
    aps = []
    
    generator = tracker.generator
    for frame_idx, frame in enumerate(tqdm(generator, desc=f"Evaluating {model_path}")):
        result_frame = tracker.callback(frame, frame_idx)
        
        results = tracker.model(frame, imgsz=1280, verbose=False)[0]
        detections = Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, tracker.CLASS_ID)]
        detections = tracker.byte_tracker.update_with_detections(detections)
        
        gt_boxes = gt_dict.get(frame_idx, [])
        
        pred_boxes = []
        if len(detections) > 0:
            for box, conf in zip(detections.xyxy, detections.confidence):
                pred_boxes.append([box[0], box[1], box[2], box[3], conf])
        
        ap = calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.5)
        aps.append(ap)
        
        matched_gt = set()
        for pred_box in pred_boxes:
            max_iou = 0
            matched_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box[:4], gt_box)
                if iou > max_iou and gt_idx not in matched_gt:
                    max_iou = iou
                    matched_idx = gt_idx
            
            if max_iou > 0.5:
                tp_count += 1
                matched_gt.add(matched_idx)
            else:
                fp_count += 1
        
        fn_count += len(gt_boxes) - len(matched_gt)
        total_gt_count += len(gt_boxes)
        
        frame_preds = [1] * len(pred_boxes) + [0] * (len(gt_boxes) - len(pred_boxes))
        frame_gts = [1] * len(gt_boxes)
        all_preds.extend(frame_preds[:len(frame_gts)])
        all_gts.extend(frame_gts)
    
    # Lấy in_count và out_count từ LineZone
    pred_in_count = tracker.line_zone.in_count
    pred_out_count = tracker.line_zone.out_count

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    mAP = np.mean(aps) if aps else 0
    in_count_accuracy = 1 - abs(pred_in_count - gt_in_count) / gt_in_count if gt_in_count > 0 else 0
    out_count_accuracy = 1 - abs(pred_out_count - gt_out_count) / gt_out_count if gt_out_count > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'mAP': mAP,
        'in_count_accuracy': in_count_accuracy,
        'out_count_accuracy': out_count_accuracy,
'tp': int(tp_count),  # Làm tròn
        'fp': int(fp_count),  # Làm tròn
        'fn': int(fn_count),  # Làm tròn
        'pred_in_count': int(pred_in_count),  # Làm tròn
        'pred_out_count': int(pred_out_count),  # Làm tròn
        'gt_in_count': int(gt_in_count),  # Làm tròn
        'gt_out_count': int(gt_out_count)  # Làm tròn
    }

def main():
    INPUT_VIDEO = "assets/video/MOT17-04-SDP-raw.mp4"
    LABEL_FOLDER = "train/labels"
    OUTPUT_COCO = "assets/video/MOT17-04-SDP-raw-output-coco.mp4"
    OUTPUT_YOLOV8M = "assets/video/MOT17-04-SDP-raw-output-yolov8m.mp4"
    MODEL_COCO = "bestCOCO.pt"
    MODEL_YOLOV8M = "best_yolov8m_coco.pt"
    
    print("Evaluating bestCOCO model...")
    coco_results = evaluate_model(MODEL_COCO, INPUT_VIDEO, LABEL_FOLDER, OUTPUT_COCO)
    
    print("Evaluating best_yolov8m_coco model...")
    yolov8m_results = evaluate_model(MODEL_YOLOV8M, INPUT_VIDEO, LABEL_FOLDER, OUTPUT_YOLOV8M)
    
    report = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'mAP', 'In Count Accuracy', 'Out Count Accuracy', 'TP', 'FP', 'FN', 'Predicted In Count', 'Predicted Out Count', 'Ground Truth In Count', 'Ground Truth Out Count'],
        'bestCOCO': [
            coco_results['precision'], coco_results['recall'], coco_results['mAP'],
            coco_results['in_count_accuracy'], coco_results['out_count_accuracy'],
            coco_results['tp'], coco_results['fp'], coco_results['fn'],
            coco_results['pred_in_count'], coco_results['pred_out_count'],
            coco_results['gt_in_count'], coco_results['gt_out_count']
        ],
        'best_yolov8m_coco': [
            yolov8m_results['precision'], yolov8m_results['recall'], yolov8m_results['mAP'],
            yolov8m_results['in_count_accuracy'], yolov8m_results['out_count_accuracy'],
            yolov8m_results['tp'], yolov8m_results['fp'], yolov8m_results['fn'],
            yolov8m_results['pred_in_count'], yolov8m_results['pred_out_count'],
            yolov8m_results['gt_in_count'], yolov8m_results['gt_out_count']
        ]
    })
    
    print("\nEvaluation Results:")
    print(report.to_string(index=False))
    report.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to 'model_comparison_results.csv'")

if __name__ == "__main__":
    main()