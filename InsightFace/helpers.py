from PIL import Image 
import cv2
from imutils import resize

def show_dets(image, dets, vis_thres, names=[]):
    for i, b in enumerate(dets):
        if b[4] < vis_thres:
            continue
        b = list(map(int, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        if names:
            text = "{}".format(names[i])
            cv2.putText(image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    return image

def get_faces(image, dets, vis_thres=0.6):
    size = (112, 112)
    faces = []
    for b in dets:
        if b[4] < vis_thres:
            continue
        boxes = b[0:4].astype(int)        
        # boxes = boxes[0,:]
        crop_face = image[boxes[1]:boxes[3], boxes[0]:boxes[2]]
        crop_face = cv2.resize(crop_face, size)
        faces.append(crop_face)
    return faces


def detect_n_recg(detector, recognizer, img_path):
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize(image, width=500)
    dets, _ = detector.detect_faces(image)
    
    # res1 = show_dets(np.float32(test1), dets, 0.6)
    faces = get_faces(image, dets)
    # cv2.imwrite('data/images/test.jpg', cv2.cvtColor(faces[0], cv2.COLOR_RGB2BGR))
    # cv2.imwrite('data/images/test1.jpg', cv2.cvtColor(faces[1], cv2.COLOR_RGB2BGR))

    faces_btch = recognizer.get_image_batch(faces)
    embds = recognizer.get_embeddings(faces_btch)
    results = recognizer.recognize(embds)
    final_image = show_dets(image, dets, 0.6, results)
    return image, dets, results, final_image


def detect_and_save(detector, img_path, name, vis_thres=0.6):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize(image, width=500)
    dets, _ = detector.detect_faces(image)
    # print(dets.shape)
    size = (112, 112)
    for b in dets:
        if b[4] < vis_thres:
            continue
        boxes = b[0:4].astype(int)        
        crop_face = image[boxes[1]:boxes[3], boxes[0]:boxes[2]]
        crop_face = cv2.resize(crop_face, size)
        cv2.imwrite(name, cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB))
    return image