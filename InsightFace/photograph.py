
import cv2
import argparse
import os
from retinaface.detector import RetinafaceDetector
def get_args():

    parse = argparse.ArgumentParser(description="Operation to take a picture")
    parse.add_argument('--name', default='test', type=str, help="Your name")
    parse.add_argument('--output-dir', default='data/images', type=str, help="Picture saving dir")
    args = parse.parse_args()

    return args

def main(args, detector):

    name_dir = args.output_dir + '/' + args.name
    if not os.path.exists(name_dir):
        os.mkdir(name_dir )
        print("make dir:", name_dir + '/')
        
    cam = cv2.VideoCapture(0)
    print("Open Camera.")
    num = 0
    while cam.isOpened():
        
        _, frame = cam.read()

        cv2.putText(frame, "S:Save image.", (10, 300), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Q:Quite.", (10, 330), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        flag = cv2.waitKey(1)

        cv2.imshow("Photograph", frame)

        if flag == ord('s'):
            filename = name_dir + '/' + args.name + str(num) + '.jpg'
            cv2.imwrite(filename, frame)
            print("Save image: ", filename)
            num += 1
        elif flag == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = get_args()
    detector = RetinafaceDetector(net='mnet')
    main(args, detector)