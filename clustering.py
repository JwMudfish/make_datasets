import glob
import timm
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as a
import cv2

from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from albumentations.pytorch import ToTensorV2

from pprint import pprint

class Clustering:
    def __init__(self, name):
        self.model_name = name
        self.model = self.model_load()
        self.show_cls = True
        self.batch_size = 50

    def model_load(self):
        # torch.cuda.empty_cache()
        model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        # model = model.cuda()
        model.eval()
        return model

    def pil_to_tensor(self, file):
        config = resolve_data_config({}, model=self.model)
        transform = create_transform(**config)
        img = Image.open(file).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        # tensor = tensor.cuda()
        return tensor

    @staticmethod
    def array_to_tensor(img):
        def get_transforms(img_size):
            return a.Compose([
                a.Resize(img_size, img_size),
                a.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
        tensor = get_transforms(224)(image=img)['image']
        tensor = tensor.unsqueeze(0)
        # tensor = tensor.cuda()
        return tensor

    def make_batch(self, data):
        mini_batch = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        return mini_batch

    def extract_features(self, tensor):
        with torch.no_grad():
            outputs = self.model(tensor).cpu().detach().numpy()
        return outputs

    @staticmethod
    def run_pca(features):
        plt.cla()  # Clear the current axes
        feat = np.array(features)
        feat = feat.reshape(-1, feat.shape[-1])
        pca = PCA(n_components=10, random_state=0)
        pca.fit(feat)
        x = pca.transform(feat)
        return x

    @staticmethod
    def run_kmeans(x, k):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(x)
        return kmeans

    @staticmethod
    def get_groups(images, kmeans):
        groups = {}
        for img, cluster in zip(images, list(map(int, kmeans.labels_))):
            if cluster not in groups.keys():
                groups[cluster] = []
            groups[cluster].append(img)
        return groups

    def clustering(self, images, k=10):
        print('clustering 시작')
        print('k : ',k)
        # tensor_list = [self.array_to_tensor(img) for _, img in images.items()]
        tensor_list = [self.array_to_tensor(cv2.imread(img)) for img in images]

        # batch
        batch = self.make_batch(tensor_list)
        features = np.concatenate([self.extract_features(torch.cat(mini, dim=0)) for mini in batch], axis=0)

        # ones
        # features = [self.extract_features(tensor) for tensor in tensor_list]

        x = self.run_pca(features)
        kmeans = self.run_kmeans(x, k)
        # groups = self.get_groups(list(images.values()), kmeans)
        groups = self.get_groups(images, kmeans)

        if self.show_cls:
            plt.clf()  # Clear the current axes
            y_kmeans = kmeans.predict(x)
            plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')
            centers = kmeans.cluster_centers_
            score = silhouette_score(x, kmeans.labels_, metric='euclidean')
            plt.title(f'{self.model_name}={score}')
            plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
            # plt.savefig(f'{self.model_name}.png')
            # plt.show()
        return groups

if __name__ == "__main__":
    path = '/data/1_sr_rnd/make_false_data/cass_fresh_500/0/'
    files = glob.glob(path + '*.jpg')

    cls = Clustering('resnet26')
    group = cls.clustering(files)

    pprint(group)

    import os
    import shutil
    save_path = './result/'
    for clu, img_list in group.items():
        sp = save_path + str(clu) + '/'
        os.makedirs(sp, exist_ok=True)
        for imgs in img_list:
            shutil.copy(imgs, sp)
