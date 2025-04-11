import os
import numpy as np
from ultralytics import YOLO
from object_tracking import ObjectTracking
from supervision import Detections, BoundingBoxAnnotator, LabelAnnotator
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

def load_ground_truth(label_folder):
    """Load ground truth từ folder labels với định dạng tên file frame_XXXX.jpg.rf.<hash>.txt"""
    gt_dict = {}
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
    return gt_dict

def calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Tính Average Precision (AP) tại IoU threshold"""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    
    # Sắp xếp dự đoán theo confidence (giả sử confidence giảm dần)
    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)  # [x_min, y_min, x_max, y_max, conf]
    
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
    
    # Tính precision và recall tại từng điểm
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros(len(tp))
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Tính AP bằng phương pháp 11-point interpolation
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
    # Thay BoxAnnotator bằng BoundingBoxAnnotator và LabelAnnotator
    tracker = ObjectTracking(input_video, output_video)
    tracker.model = YOLO(model_path).to("cuda")
    tracker.model.fuse()
    tracker.box_annotator = BoundingBoxAnnotator(thickness=1)
    tracker.label_annotator = LabelAnnotator(text_thickness=1, text_scale=0.3)
    
    gt_dict = load_ground_truth(label_folder)
    
    all_preds = []
    all_gts = []
    tp_count, fp_count, fn_count = 0, 0, 0
    total_gt_count = 0
    total_pred_count = 0
    aps = []  # Lưu AP cho từng frame
    
    generator = tracker.generator
    for frame_idx, frame in enumerate(tqdm(generator, desc=f"Evaluating {model_path}")):
        result_frame = tracker.callback(frame, frame_idx)
        
        results = tracker.model(frame, imgsz=1280, verbose=False)[0]
        detections = Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, tracker.CLASS_ID)]
        detections = tracker.byte_tracker.update_with_detections(detections)
        
        gt_boxes = gt_dict.get(frame_idx, [])
        
        # Chuẩn bị pred_boxes với confidence
        pred_boxes = []
        if len(detections) > 0:
            for box, conf in zip(detections.xyxy, detections.confidence):
                pred_boxes.append([box[0], box[1], box[2], box[3], conf])
        
        # Tính AP cho frame này
        ap = calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.5)
        aps.append(ap)
        
        # Tính TP, FP, FN
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
        total_pred_count += len(pred_boxes)  # Đếm tổng số đối tượng được phát hiện
        
        frame_preds = [1] * len(pred_boxes) + [0] * (len(gt_boxes) - len(pred_boxes))
        frame_gts = [1] * len(gt_boxes)
        all_preds.extend(frame_preds[:len(frame_gts)])
        all_gts.extend(frame_gts)
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    mAP = np.mean(aps) if aps else 0
    counting_accuracy = 1 - abs(total_pred_count - total_gt_count) / total_gt_count if total_gt_count > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'mAP': mAP,
        'counting_accuracy': counting_accuracy,
        'tp': tp_count,
        'fp': fp_count,
        'fn': fn_count,
        'total_pred_count': total_pred_count,
        'total_gt_count': total_gt_count
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
        'Metric': ['Precision', 'Recall', 'mAP', 'Counting Accuracy', 'TP', 'FP', 'FN', 'Predicted Count', 'Ground Truth Count'],
        'bestCOCO': [
            coco_results['precision'], coco_results['recall'], coco_results['mAP'],
            coco_results['counting_accuracy'], coco_results['tp'], coco_results['fp'],
            coco_results['fn'], coco_results['total_pred_count'], coco_results['total_gt_count']
        ],
        'best_yolov8m_coco': [
            yolov8m_results['precision'], yolov8m_results['recall'], yolov8m_results['mAP'],
            yolov8m_results['counting_accuracy'], yolov8m_results['tp'], yolov8m_results['fp'],
            yolov8m_results['fn'], yolov8m_results['total_pred_count'], yolov8m_results['total_gt_count']
        ]
    })
    
    print("\nEvaluation Results:")
    print(report.to_string(index=False))
    report.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to 'model_comparison_results.csv'")

if __name__ == "__main__":
    main()