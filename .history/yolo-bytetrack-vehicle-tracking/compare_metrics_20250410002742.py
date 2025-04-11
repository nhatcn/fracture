import numpy as np
import motmetrics as mm
from sklearn.metrics import precision_score, recall_score, f1_score

# ------------------- Hàm đọc kết quả counting từ file -------------------
def read_counting_results(counting_file):
    counting_results = {}
    with open(counting_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(": ")
            counting_results[key] = int(value)
    return counting_results["in_count"], counting_results["out_count"]

# ------------------- Hàm đọc ground truth counting từ file -------------------
def read_gt_counting_results(gt_counting_file):
    gt_counting_results = {}
    with open(gt_counting_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(": ")
            gt_counting_results[key] = int(value)
    return gt_counting_results["gt_in_count"], gt_counting_results["gt_out_count"]

# ------------------- Hàm tính detection metrics -------------------
def compute_detection_metrics(gt_file, detection_file, model_name, iou_threshold=0.5):
    # Đọc ground truth
    gt = []
    with open(gt_file, 'r') as f:
        for line in f:
            frame, obj_id, x, y, w, h, conf, _, _, _ = map(float, line.strip().split(','))
            if conf > 0:
                gt.append([int(frame), x, y, w, h])

    # Đọc kết quả detection
    detections = []
    with open(detection_file, 'r') as f:
        for line in f:
            frame, x, y, w, h, conf, class_id = map(float, line.strip().split(','))
            detections.append([int(frame), x, y, w, h, conf])

    # Tạo danh sách ground truth và predictions theo frame
    gt_frames = {}
    det_frames = {}
    for frame, x, y, w, h in gt:
        if frame not in gt_frames:
            gt_frames[frame] = []
        gt_frames[frame].append([x, y, w, h])

    for frame, x, y, w, h, conf in detections:
        if frame not in det_frames:
            det_frames[frame] = []
        det_frames[frame].append([x, y, w, h, conf])

    # Tính TP, FP, FN
    tp, fp, fn = 0, 0, 0
    y_true, y_pred = [], []

    for frame in sorted(set(gt_frames.keys()).union(set(det_frames.keys()))):
        gt_boxes = gt_frames.get(frame, [])
        det_boxes = det_frames.get(frame, [])

        gt_matched = [False] * len(gt_boxes)
        det_matched = [False] * len(det_boxes)

        if gt_boxes and det_boxes:
            gt_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in gt_boxes]
            det_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in det_boxes]
            distances = mm.distances.iou_matrix(gt_boxes, det_boxes, max_iou=1 - iou_threshold)

            for gt_idx in range(len(gt_boxes)):
                if gt_idx >= distances.shape[0]:
                    continue
                min_dist_idx = np.argmin(distances[gt_idx])
                if min_dist_idx < distances.shape[1] and distances[gt_idx, min_dist_idx] < 1 - iou_threshold:
                    gt_matched[gt_idx] = True
                    det_matched[min_dist_idx] = True

        tp += sum(1 for matched in gt_matched if matched)
        fn += sum(1 for matched in gt_matched if not matched)
        fp += sum(1 for matched in det_matched if not matched)

        y_true.extend([1] * len(gt_boxes))
        y_pred.extend([1] * sum(det_matched) + [0] * sum(1 for matched in det_matched if not matched))

    # Tính precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

# ------------------- Hàm tính tracking metrics -------------------
def compute_tracking_metrics(gt_file, tracking_file, model_name):
    # Đọc ground truth
    gt = []
    with open(gt_file, 'r') as f:
        for line in f:
            frame, obj_id, x, y, w, h, conf, _, _, _ = map(float, line.strip().split(','))
            if conf > 0:
                gt.append([int(frame), int(obj_id), x, y, w, h])

    # Đọc kết quả tracking
    tracking_results = []
    with open(tracking_file, 'r') as f:
        for line in f:
            frame, obj_id, x, y, w, h = map(float, line.strip().split(','))
            tracking_results.append([int(frame), int(obj_id), x, y, w, h])

    # Tạo accumulator để tính metrics
    acc = mm.MOTAccumulator(auto_id=True)

    # Chuyển đổi tracking results thành định dạng phù hợp
    pred_frames = {}
    for frame, obj_id, x, y, w, h in tracking_results:
        if frame not in pred_frames:
            pred_frames[frame] = []
        pred_frames[frame].append([obj_id, x, y, w, h])

    # Chuyển đổi ground truth thành định dạng phù hợp
    gt_frames = {}
    for frame, obj_id, x, y, w, h in gt:
        if frame not in gt_frames:
            gt_frames[frame] = []
        gt_frames[frame].append([obj_id, x, y, w, h])

    # Tính metrics cho từng frame
    for frame in sorted(set(gt_frames.keys()).union(set(pred_frames.keys()))):
        gt_objs = gt_frames.get(frame, [])
        pred_objs = pred_frames.get(frame, [])

        gt_ids = [obj[0] for obj in gt_objs]
        pred_ids = [obj[0] for obj in pred_objs]

        gt_boxes = [[obj[2], obj[3], obj[2] + obj[4], obj[3] + obj[5]] for obj in gt_objs]  # [x1, y1, x2, y2]
        pred_boxes = [[obj[1], obj[2], obj[1] + obj[3], obj[2] + obj[4]] for obj in pred_objs]

        if gt_boxes and pred_boxes:
            distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            distances = np.array([])

        acc.update(gt_ids, pred_ids, distances)

    # Tính toán metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'num_switches', 'fp', 'fn'], name=model_name)
    return summary

# ------------------- Hàm tính counting metrics -------------------
def compute_counting_metrics(gt_in_count, gt_out_count, in_count, out_count, model_name):
    missed_in = gt_in_count - in_count
    missed_out = gt_out_count - out_count
    return {
        "model": model_name,
        "in_count": in_count,
        "out_count": out_count,
        "net_count": in_count - out_count,
        "missed_in": missed_in,
        "missed_out": missed_out
    }

# ------------------- Main Function -------------------
if __name__ == "__main__":
    # Đường dẫn đến ground truth và kết quả của hai mô hình
    gt_file = "pedestrian_video_gt.txt"  # File ground truth ở định dạng MOTChallenge
    gt_counting_file = "gt_counting.txt"  # File chứa ground truth counting
    yolov8m_detection_file = "best7_detections.txt"
    yolov8m_tracking_file = "best7_tracking.txt"
    yolov8m_counting_file = "best7_counting.txt"
    yolov8m_cbam_detection_file = "yolov8m_resblock_cbam_detections.txt"
    yolov8m_cbam_tracking_file = "yolov8m_resblock_cbam_tracking.txt"
    yolov8m_cbam_counting_file = "yolov8m_resblock_cbam_counting.txt"

    # Đọc ground truth counting từ file
    gt_in_count, gt_out_count = read_gt_counting_results(gt_counting_file)
    print(f"Ground Truth Counting: In = {gt_in_count}, Out = {gt_out_count}")

    # Đọc kết quả counting từ file
    yolov8m_in_count, yolov8m_out_count = read_counting_results(yolov8m_counting_file)
    yolov8m_cbam_in_count, yolov8m_cbam_out_count = read_counting_results(yolov8m_cbam_counting_file)

    # Tính detection metrics
    yolov8m_det_metrics = compute_detection_metrics(gt_file, yolov8m_detection_file, "YOLOv8m")
    yolov8m_cbam_det_metrics = compute_detection_metrics(gt_file, yolov8m_cbam_detection_file, "YOLOv8m_CBAM")

    # Tính tracking metrics
    yolov8m_track_metrics = compute_tracking_metrics(gt_file, yolov8m_tracking_file, "YOLOv8m")
    yolov8m_cbam_track_metrics = compute_tracking_metrics(gt_file, yolov8m_cbam_tracking_file, "YOLOv8m_CBAM")

    # Tính counting metrics
    yolov8m_count_metrics = compute_counting_metrics(gt_in_count, gt_out_count, yolov8m_in_count, yolov8m_out_count, "YOLOv8m")
    yolov8m_cbam_count_metrics = compute_counting_metrics(gt_in_count, gt_out_count, yolov8m_cbam_in_count, yolov8m_cbam_out_count, "YOLOv8m_CBAM")

    # In kết quả so sánh
    print("\n=== Detection Metrics Comparison ===")
    print(f"{'Model':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<8} {'FP':<8} {'FN':<8}")
    print(f"{'-'*70}")
    print(f"{yolov8m_det_metrics['model']:<15} {yolov8m_det_metrics['precision']:<10.3f} {yolov8m_det_metrics['recall']:<10.3f} {yolov8m_det_metrics['f1']:<10.3f} {yolov8m_det_metrics['tp']:<8} {yolov8m_det_metrics['fp']:<8} {yolov8m_det_metrics['fn']:<8}")
    print(f"{yolov8m_cbam_det_metrics['model']:<15} {yolov8m_cbam_det_metrics['precision']:<10.3f} {yolov8m_cbam_det_metrics['recall']:<10.3f} {yolov8m_cbam_det_metrics['f1']:<10.3f} {yolov8m_cbam_det_metrics['tp']:<8} {yolov8m_cbam_det_metrics['fp']:<8} {yolov8m_cbam_det_metrics['fn']:<8}")

    print("\n=== Tracking Metrics Comparison ===")
    print(yolov8m_track_metrics)
    print(yolov8m_cbam_track_metrics)

    print("\n=== Counting Metrics Comparison ===")
    print(f"{'Model':<15} {'In Count':<10} {'Out Count':<10} {'Net Count':<10} {'Missed In':<10} {'Missed Out':<10}")
    print(f"{'-'*70}")
    print(f"{yolov8m_count_metrics['model']:<15} {yolov8m_count_metrics['in_count']:<10} {yolov8m_count_metrics['out_count']:<10} {yolov8m_count_metrics['net_count']:<10} {yolov8m_count_metrics['missed_in']:<10} {yolov8m_count_metrics['missed_out']:<10}")
    print(f"{yolov8m_cbam_count_metrics['model']:<15} {yolov8m_cbam_count_metrics['in_count']:<10} {yolov8m_cbam_count_metrics['out_count']:<10} {yolov8m_cbam_count_metrics['net_count']:<10} {yolov8m_cbam_count_metrics['missed_in']:<10} {yolov8m_cbam_count_metrics['missed_out']:<10}")