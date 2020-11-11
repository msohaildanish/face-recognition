import cv2
import numpy as np
from InsightFace.retinaface.detector import RetinafaceDetector
from InsightFace.helpers import show_dets, get_faces
from imutils import resize
import base64
import os
# from PIL import Image
# from StringIO import StringIO

def upload_and_detec(file, path):
    detector = RetinafaceDetector(net='mnet', weights_path='InsightFace/retinaface/weights/mobilenet0.25_Final.pth')
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    dets, _ = detector.detect_faces(image)
    faces = get_faces(image, dets)
    cv2.imwrite(path, faces[0])
    # ret, jpeg = cv2.imencode('.jpg', frame)
    
    
    
def upload_and_detec1(images, name):
    detector = RetinafaceDetector(net='mnet', weights_path='InsightFace/retinaface/weights/mobilenet0.25_Final.pth')
    path = 'InsightFace/data/images/faces/' + name
    if not os.path.exists(path):
        os.makedirs(path)
    for i, image in enumerate(images):
        
        file_name = "{}/{}_{}.jpg".format(path, name, i)
        image = data_uri_to_cv2_img(image)
        dets, _ = detector.detect_faces(image)
        faces = get_faces(image, dets)
        cv2.imwrite(file_name, faces[0])
    
    
def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img