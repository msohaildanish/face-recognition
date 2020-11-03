import cv2
from imutils import resize
import time
import sys
sys.path.append('InsightFace')
from InsightFace.retinaface.detector import RetinafaceDetector
# from  import RetinafaceDetector
from InsightFace.helpers import show_dets, get_faces
from InsightFace.recognizer import Recognizer

class VideoCamera(object):
    def __init__(self):
        self.detector = RetinafaceDetector(net='mnet', weights_path='InsightFace/retinaface/weights/mobilenet0.25_Final.pth')
        self.recognizer = Recognizer('InsightFace/data/database', weights_path='InsightFace/insight-face-v3.pt')
        self.video = cv2.VideoCapture(0)
        time.sleep(1.0)

    def __del__(self):
        self.video.release()
    
    def get_frame(self, dets, names):
        success, frame = self.video.read()
        frame = resize(frame, width=700)
        frame = show_dets(frame, dets, 0.6, names)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    
    def detect_face(self):
        success, frame = self.video.read()
        frame = resize(frame, width=700)
        dets, _ = self.detector.detect_faces(frame)
        frame = show_dets(frame, dets, 0.6, [])
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), dets
    
    def recognize_face(self):
        success, frame = self.video.read()
        frame = resize(frame, width=700)
        dets, _ = self.detector.detect_faces(frame)
        faces = get_faces(frame, dets)
        names = self.recognizer.recognize_frame(faces)
        frame = show_dets(frame, dets, 0.6, names)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), dets, names