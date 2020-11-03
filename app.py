from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def detect(camera):
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
        #     frame = camera.get_frame(dets, names)
        # counter += 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        

@app.route('/video_feed')
def video_feed():
    video_camera = VideoCamera()
    return Response(detect(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)