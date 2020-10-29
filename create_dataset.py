import cv2
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

def list_dir(path):

    path_list = []
    name_list = []

    for root, branch, leaf in os.walk(path):
        if len(branch) != 0:
            continue
        name = root.split('/')[-1]
        name_list.extend([name] * len(leaf))
        path_list.extend(os.path.join(root, filename) for filename in leaf)

    return name_list, path_list


def save_img(save_dir, face_img):
    plt.imsave(save_dir, face_img)

def crop_face(args):
    
    img_dir = args.input
    out_dir = args.output
    size = args.size.split(',')
    size = (int(size[0]), int(size[1]))

    name_list, path_list = list_dir(img_dir)
    mtcnn_model = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    face_detect = MtcnnDetector(model_folder=mtcnn_model, ctx=mx.gpu(args.gpu), num_worker=1, accurate_landmark = True)
    face_dir = []
    for name, path in zip(name_list, path_list):
        img = cv2.imread(path)
        try:
            boxes, _ = face_detect.detect_face(img, det_type=0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            boxes = boxes[:,0:4].astype(int)        
            boxes = boxes[0,:]
            crop_face = img[boxes[1]:boxes[3], boxes[0]:boxes[2]]
            crop_face = cv2.resize(crop_face, size)
            save_dir = os.path.join(out_dir, name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            basename = os.path.basename(path)
            save_dir = os.path.join(save_dir, basename)
            save_img(save_dir, crop_face)
            print("save image in ", save_dir)
            face_dir.append(save_dir)
        except:
            pass
    
    return face_dir

def save_as_npy(face_list, args):
    
    # 加载facemodel
    model = FaceModel(args)
    # 两个列表分别装着图片结果和对应标签
    embedding_data = []
    embedding_label = []
    for face in face_list:
        img = cv2.imread(face)
        result = model.get_input(img)
        if result:
            img, _ = result
            img = model.get_feature(img)
            name = os.path.split(os.path.split(face)[0])[1]
            embedding_data.append(img)
            embedding_label.append(name)

    # 只用将list对象转换为np.array对象的时候才能被np.save保存
    embedding_data = np.array(embedding_data)
    embedding_label = np.array(embedding_label)
    #print(embedding_data)
    #print(embedding_label)
    np.save(args.data_path, embedding_data)
    np.save(args.label_path, embedding_label)
    print("save in ", args.data_path, args.label_path)


    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Operation to create database")
    parser.add_argument('--input', default='../datasets/image/', type=str, help="input image dir(subdirs are allowed)")
    parser.add_argument('--output', default='../datasets/database/', type=str, help="output database dir")
    parser.add_argument('--size', default="112,112", type=str, help="corp image size")
    parser.add_argument('--create-npy','-c', action='store_true', help="whether convert to database.npy save?")
    parser.add_argument('--model', default='../models/model-y1-test2/model,0000', help='path to load model.')
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--data-path', default="../datasets/npy/datas.npy", type=str, help="the path of datas.npy")
    parser.add_argument('--label-path', default="../datasets/npy/labels.npy", type=str, help="the path of labels.npy")
    args = parser.parse_args()
    face_list = crop_face(args)

    if args.create_npy:
        save_as_npy(face_list, args)
        


