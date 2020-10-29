import os
import numpy as np
import torch
# from torch import nn
from PIL import Image, ImageOps
# import argparse
from config import device
from models import resnet101
# from utils import parse_args, walkdir
# from utils import walkdir
from data_gen import data_transforms

def walkdir(folder, ext):
    # Walk through each files in a directory
    for dirpath, dirs, files in os.walk(folder):
        for filename in [f for f in files if f.lower().endswith(ext)]:
            yield os.path.abspath(os.path.join(dirpath, filename))

class Recognizer(object):
    
    def __init__(self, database_path, mode='val', threshold=1.05):
        self.database_path = database_path
        self.model = self.load_model()
        self.database = self.load_database()
        self.transformer = data_transforms[mode]
        self.threshold = threshold
        self.model.eval()
        
    def load_model(self):
        model = resnet101()
        model.load_state_dict(torch.load('insight-face-v3.pt',  map_location=torch.device(device)))
        # model = nn.DataParallel(model)
        model = model.to(device)
        return model
    
    def get_image(self, filepath, flip=False, unsqueeze=False):
        img = Image.open(filepath)
        if flip:
            img = ImageOps.flip(img)
        img = self.transformer(img)
        if unsqueeze:
            img = img.unsqueeze(0)
        return img.to(device)
    
    def load_database(self):
        
        embeddings, labels = [], []
        embeddings_path = self.database_path + '/' + 'embeddings.npy'
        names_path = self.database_path + '/' + 'names.npy'
        if os.path.exists(embeddings_path):
            embeddings = list(np.load(embeddings_path, allow_pickle=True))
            labels = list(np.load(names_path,  allow_pickle=True))
            
        return embeddings, labels
    
    def update_database(self):
        embeddings_path = self.database_path + '/' + 'embeddings.npy'
        names_path = self.database_path + '/' + 'names.npy'
        np.save(embeddings_path, self.database[0])
        np.save(names_path, self.database[1])
        print("save in ", embeddings_path, names_path)
        
    def get_embeddings(self, images):
        with torch.no_grad():
            embeddings = self.model(images.to(device))
            embeddings = embeddings.cpu().numpy()
            embeddings = embeddings/np.linalg.norm(embeddings)
            return embeddings
        
    def gen_batch_database(self, path, batch_size=40):
        print('gen features {}...'.format(path))
        # Preprocess the total files count
        files = []
        for filepath in walkdir(path, ('.jpg', '.png')):
            files.append(filepath)
        file_count = len(files)
        new_names = []
        new_embeddings = []
        with torch.no_grad():
            for start_idx in range(0, file_count, batch_size):
                end_idx = min(file_count, start_idx + batch_size)
                length = end_idx - start_idx

                imgs_0 = torch.zeros([length, 3, 112, 112], dtype=torch.float, device=device)
                for idx in range(0, length):
                    i = start_idx + idx
                    filepath = files[i]
                    imgs_0[idx] = self.get_image(filepath, flip=False)

                features_0 = self.model(imgs_0.to(device))
                features_0 = features_0.cpu().numpy()

                imgs_1 = torch.zeros([length, 3, 112, 112], dtype=torch.float, device=device)
                for idx in range(0, length):
                    i = start_idx + idx
                    filepath = files[i]
                    imgs_1[idx] = self.get_image(filepath, flip=True)

                features_1 = self.model(imgs_1.to(device))
                features_1 = features_1.cpu().numpy()

                for idx in range(0, length):
                    i = start_idx + idx
                    name = os.path.split(os.path.split(files[i])[0])[1]
                    filepath = files[i]
                    feature = features_0[idx] + features_1[idx]
                    new_embeddings.append(feature / np.linalg.norm(feature))
                    new_names.append(name)
        all_embeddings, all_names = self.database
        all_embeddings += new_embeddings
        all_names += new_names
        self.database = all_embeddings, all_names
        self.update_database()
        
    def recognize(self, faces):
        print('_______REC______')
        embeddings, labels = self.database
        name_list = []
        for face in faces:
            dist_this = []
            for embd in embeddings:
                # dist = np.sum(np.square(face - embd)) ** 0.5
                dist = ((embd - face) ** 2).sum() ** 0.5
                dist_this.append(dist)
                print(dist)
            min_dist = min(dist_this)
            number = dist_this.index(min_dist)
            print(labels[number], min_dist)
            if min_dist < self.threshold:
                name = labels[number]
            else:
                name = 'Unknow'
            name_list.append(name)

        return name_list
        
    def recognize_single_face(self, image_path):
        image = self.get_image(image_path, flip=False, unsqueeze=True)
        embeddings = self.get_embeddings(image)
        return self.recognize([embeddings])
    
    def get_image_batch(self, images):
        imgs = torch.zeros([len(images), 3, 112, 112], dtype=torch.float, device=device)
        for i, face in enumerate(images):
            imgs[i] = self.transformer(face)
        return imgs.to(device)
            
    




