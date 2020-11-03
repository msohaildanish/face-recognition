import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import argparse
from config import device, grad_clip, print_freq, num_workers, logger
from data_gen import ArcFaceDataset
from focal_loss import FocalLoss
from megaface_eval import megaface_test
from models import resnet18, resnet34, resnet50, resnet101, resnet152, ArcMarginModel
from utils import parse_args, save_checkpoint, AverageMeter, accuracy, clip_gradient


model = resnet101(args)

model.load_state_dict(torch.load('insight-face-v3.pt'))

model = nn.DataParallel(model)
model = model.to(device)


def get_image(transformer, filepath, flip=False, unsqueeze=True):
    img = Image.open(filepath)
    if flip:
        img = ImageOps.flip(img)
    img = transformer(img)
    img = img.unsqueeze(0)
    return img.to(device)


model.eval()


def load_database_by_npy(data_path, labels_path):
    
    datas = list(np.load(data_path, allow_pickle=True))
    labels = list(np.load(labels_path,  allow_pickle=True))

    return datas, labels


def recognition(imgs, database, threshold):
    labels = database[1]
    datas = database[0]
    name_list = []
    for img in imgs:
        dist_this = []
        for data in datas:
            dist = np.sum(np.square(img - data)) ** 0.5
            dist_this.append(dist)
            print(dist)
        min_dist = min(dist_this)
        number = dist_this.index(min_dist)
        print(labels[number], min_dist)
        if min_dist < threshold:
            name = labels[number]
        else:
            name = 'Unknow'
        name_list.append(name)

    return name_list


def write_npy(embedding_data, embedding_label, data_path, label_path):
    np.save(data_path, embedding_data)
    np.save(label_path, embedding_label)
    print("save in ", data_path, label_path)

def gen_database(path, model, data_path, label_path):
    model.eval()
    print('gen features {}...'.format(path))
    # Preprocess the total files count
    files = []
    for filepath in walkdir(path, ('.jpg', '.png')):
        files.append(filepath)
    file_count = len(files)
    all_names = []
    all_faces = []
    transformer = data_transforms['val']

    batch_size = 128

    with torch.no_grad():
        for start_idx in range(0, file_count, batch_size):
            end_idx = min(file_count, start_idx + batch_size)
            length = end_idx - start_idx

            imgs_0 = torch.zeros([length, 3, 112, 112], dtype=torch.float, device=device)
            for idx in range(0, length):
                i = start_idx + idx
                filepath = files[i]
                imgs_0[idx] = get_image(transformer, filepath, flip=False)

            features_0 = model(imgs_0.to(device))
            features_0 = features_0.cpu().numpy()

            imgs_1 = torch.zeros([length, 3, 112, 112], dtype=torch.float, device=device)
            for idx in range(0, length):
                i = start_idx + idx
                filepath = files[i]
                imgs_1[idx] = get_image(transformer, filepath, flip=True)

            features_1 = model(imgs_1.to(device))
            features_1 = features_1.cpu().numpy()

            for idx in range(0, length):
                i = start_idx + idx
                name = os.path.split(os.path.split(files[i])[0])[1]
                filepath = files[i]
                feature = features_0[idx] + features_1[idx]
                all_faces.append(feature / np.linalg.norm(feature))
                all_names.append(name)
    
    write_npy(all_faces, all_names, data_path, label_path)
                


