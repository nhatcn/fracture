import cv2
import os

# Đường dẫn đến video
video_path = 'fpt.mp4'

# Thư mục để lưu các frame
output_folder = 'frame'
os.makedirs(output_folder, exist_ok=True)

# Đọc video
cap = cv2.VideoCapture(video_path)

frame_count = 0
success = True

while success:
    success, frame = cap.read()
    if success:
        # Đặt tên file cho mỗi frame
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:05d}.jpg')
        # Lưu frame thành ảnh
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

cap.release()
print(f"✅ Đã trích xuất {frame_count} frame vào thư mục '{output_folder}'")
