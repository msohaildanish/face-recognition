import os
from flask import Flask, jsonify, request, render_template, redirect, url_for, flash, Response
from camera import VideoCamera
import cv2
from utils import upload_and_detec, data_uri_to_cv2_img, upload_and_detec1
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/photograph')
def photograph():
    return render_template('photograph.html')

@app.route('/upload_faces', methods=["POST", "GET"])
def upload_faces():
    error = None
    if request.method == "POST":
        file = request.files['image']
        name = request.form['person_name']
        counter = '_0'
        path = 'InsightFace/data/images/faces/' + name
        if not os.path.exists(path):
            os.makedirs(path)
            
        file_name = path + '/' + name + counter + '.jpg'
        # Read image
        image = upload_and_detec(file, file_name)
        # flash('You were successfully logged in')
    return render_template('upload_face.html', error=error)

@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.get_json()
    # print(data)
    name = data['name']
    images = data['image']
    upload_and_detec1(images, name)
    return Response('success')
    
    
def recognize(camera):
    # counter = 1
    while True:
        # dets = []
        # names = []
        # frame = None
        frame, dets, names = camera.recognize_face()
            # frame, dets = camera.detect_face()
            # faces = get_faces(frame, dets)
            # faces_btch = recognizer.get_image_batch(faces)
            # embds = recognizer.get_embeddings(faces_btch)
            # names = recognizer.recognize(embds)
        # else:
        #     print(names)
        # frame = camera.get_frame()
        # counter += 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_demo')
def video_demo():
    video_camera = VideoCamera(photograph=False)
    return Response(recognize(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, ssl_context='adhoc')