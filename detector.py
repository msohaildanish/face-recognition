from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from IPython import display
import time



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

class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.
        
        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.
        
        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]
        print(frames.shape)             
        boxes, probs = self.mtcnn.detect(frames[::self.stride])

        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                faces.append(frame[box[1]:box[3], box[0]:box[2]])
        
        return faces
    


class FaceDetector(object):
    
    def __init__(self, resize=1):

        self.model = self.load_model()
        print('Finished loading model!')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # net = net.to(device)

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def load_model(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(device))
        # fast_mtcnn = FastMTCNN(
        #     stride=4,
        #     resize=1,
        #     margin=14,
        #     factor=0.6,
        #     keep_all=True,
        #     device=device
        # )
        # return fast_mtcnn
        mtcnn = MTCNN(keep_all=True, device=device)
        return mtcnn
    
    def detect_from_image(self, image_path, save_path=''):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        dets, _ = self.model.detect(image)
        img = show_dets(image, dets)
        cv2.imwrite(save_path, img)
    
    def detect(self, image):

        faces, _ = self.model(image)
        # show image
        return faces










detector = FaceDetector()
detector.detect_from_image('crowd.jpg', 'crowd_result1.jpg')
