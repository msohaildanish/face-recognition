import torch
import numpy as np
import cv2
import time
import argparse
# from InsightFace_PyTorch.utils import load_model
from imutils.video import FPS
from imutils import resize
from helpers import get_faces, show_dets

from retinaface.detector import RetinafaceDetector
from recognizer import Recognizer

    


# def show_dets(image, dets, vis_thres):
#     for b in dets:
#         if b[4] < vis_thres:
#             continue
#         text = "{:.1f}".format(b[4])
#         b = list(map(int, b))
#         cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
#         cx = b[0]
#         cy = b[1] + 12
#         cv2.putText(image, text, (cx, cy),
#                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
#     return image

def cv_stream(detector, recognizer):
    print("[INFO] starting video file thread...")
    video = cv2.VideoCapture(0)
    time.sleep(1.0)
    # start the FPS timer
    fps = FPS().start()
    counter = 1
    dets = []
    names = []
    # loop over frames from the video file stream
    while(True):
        # grab the frame from the threaded video file stream, resize
        success, frame = video.read()
        frame = resize(frame, width=700)
        if counter % 3 == 0:
            dets, _ = detector.detect_faces(frame)
            faces = get_faces(frame, dets)
            faces_btch = recognizer.get_image_batch(faces)
            embds = recognizer.get_embeddings(faces_btch)
            names = recognizer.recognize(embds)
        frame = show_dets(frame, dets, 0.6, names)
        # show the frame and update the FPS counter
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()
        counter += 1
    
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # imutils_stream()
    # detector, recog = load_models()
    detector = RetinafaceDetector(net='mnet')
    recognizer = Recognizer('data/database')
    cv_stream(detector, recognizer)
    