import numpy as np
import cv2
import torch
from imutils.video import FileVideoStream
from imutils.video import FPS
from imutils import resize
import time

from facenet_pytorch import MTCNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all=True, device=device)


def imutils_stream():
    # start the file video stream thread and allow the buffer to
    # start to fill
    print("[INFO] starting video file thread...")
    fvs = FileVideoStream(0).start()
    time.sleep(1.0)
    # start the FPS timer
    fps = FPS().start()

    # loop over frames from the video file stream
    
    while fvs.more():
        # grab the frame from the threaded video file stream, resize
        frame = fvs.read()
        frame = resize(frame, width=700)
        frame = detector.detect(frame)
        # show the frame and update the FPS counter
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    fvs.stop()
    
    
def show_dets(image, dets):
    for b in dets:
        
        # text = "{:.1f}".format(b[4])
        b = b.tolist()
        b = list(map(int, b))
        
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        # cx = b[0]
        # cy = b[1] + 12
        # cv2.putText(image, text, (cx, cy),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    return image


def cv_stream():
    print("[INFO] starting video file thread...")
    video = cv2.VideoCapture(0)
    time.sleep(1.0)
    # start the FPS timer
    fps = FPS().start()
    counter = 1
    dets = []
    # loop over frames from the video file stream
    while(True):
        # grab the frame from the threaded video file stream, resize
        success, frame = video.read()
        frame = resize(frame, width=700)
        
        # if counter % 3 == 0:
        dets, _ = mtcnn.detect(frame)
        frame = show_dets(frame, dets)
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
    cv_stream()
    