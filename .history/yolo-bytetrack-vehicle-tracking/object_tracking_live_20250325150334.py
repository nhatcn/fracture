from object_tracking import ObjectTracking
from ultralytics import YOLO
import cv2
import yt_dlp
from supervision import Point, LineZone, BoxAnnotator, TraceAnnotator, LineZoneAnnotator
from bytetrack.byte_track import ByteTrack

class ObjectTrackingLive(ObjectTracking):
    def __init__(self, input_url):

        print(f"Fetching stream from: {input_url}")
        ydl_opts = {
            'format': 'best[ext=mp4]', 
            'quiet': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(input_url, download=False)
                stream_url = info['url']  
        except Exception as e:
            raise ValueError(f"Failed to fetch stream URL: {str(e)}")
        
        self.capture = cv2.VideoCapture(stream_url)  
        

        if not self.capture.isOpened():
            raise ValueError("Unable to open YouTube Live stream. Check URL or network connection.")
        

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS)) or 30  
        print(f"Stream info: {self.width}x{self.height}, FPS: {self.fps}")
        

        self.model = YOLO("yolov8m.pt").to("cuda")
        self.model.fuse()
        print("Model loaded successfully")
        
        self.CLASS_NAMES_DICT = self.model.model.names
        self.CLASS_ID = [0]  # Person or desired class index

        self.LINE_START = Point( 0,int(self.height * 0.5))
        self.LINE_END = Point( self.width, int(self.height * 0.99))

        self.byte_tracker = ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=self.fps
        )

        self.line_zone = LineZone(start=self.LINE_START, end=self.LINE_END)
        self.box_annotator = BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.6)
        self.trace_annotator = TraceAnnotator(thickness=2, trace_length=50)
        self.line_zone_annotator = LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.8)

    def process(self):
        print("Starting realtime tracking. Press 'q' to exit.")
        frame_index = 0
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                print("End of stream or error occurred.")
                break

            print(f"Processing frame {frame_index}")
            result_frame = self.callback(frame, frame_index)

            cv2.imshow("Realtime Tracking", result_frame)
            
            # Thoát khi nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user.")
                break

            frame_index += 1

        self.capture.release()
        cv2.destroyAllWindows()  
        print("Realtime tracking finished.")