import glob
import timm
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as a

from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from albumentations.pytorch import ToTensorV2


class Clustering:
    def __init__(self, name):
        self.model_name = name
        self.model = self.model_load()
        self.show_cls = False
        self.batch_size = 50

    def model_load(self):
        torch.cuda.empty_cache()
        model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        model = model.cuda()
        model.eval()
        return model

    def pil_to_tensor(self, file):
        config = resolve_data_config({}, model=self.model)
        transform = create_transform(**config)
        img = Image.open(file).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        tensor = tensor.cuda()
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
        tensor = tensor.cuda()
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
        tensor_list = [self.array_to_tensor(img) for _, img in images.items()]

        # batch
        batch = self.make_batch(tensor_list)
        features = np.concatenate([self.extract_features(torch.cat(mini, dim=0)) for mini in batch], axis=0)

        # ones
        # features = [self.extract_features(tensor) for tensor in tensor_list]

        x = self.run_pca(features)
        kmeans = self.run_kmeans(x, k)
        groups = self.get_groups(list(images.values()), kmeans)

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
        return groups, plt

    def graph(self):
        aa = 10
        # plt.plot(np.cumsum(pp.explained_variance_ratio_))
        # plt.xlabel('number of components')
        # plt.ylabel('cumulative explained variance')
        # plt.show()

        # import matplotlib.pyplot as plt
        # plt.rcParams["figure.figsize"] = (12, 6)
        #
        # fig, ax = plt.subplots()
        # xi = np.arange(1, pca.shape[1] + 1, step=1)
        # y = np.cumsum(pp.explained_variance_ratio_)
        #
        # plt.ylim(0.0, 1.1)
        # plt.plot(xi, y, marker='o', linestyle='--', color='b')
        #
        # plt.xlabel('Number of Components')
        # plt.xticks(np.arange(0, pca.shape[1] + 1, step=1))  # change from 0-based array index to 1-based human-readable label
        # plt.ylabel('Cumulative variance (%)')
        # plt.title('The number of components needed to explain variance')
        #
        # plt.axhline(y=0.95, color='r', linestyle='-')
        # plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)
        #
        # ax.grid(axis='x')
        # plt.show()


if __name__ == "__main__":
    path = '/home/interminds/Desktop/cass_fresh_500/'
    files = glob.glob(path + '*.jpg')

    cls = Clustering('resnet26')
    group = cls.clustering(files)

    import os
    import shutil
    save_path = '/home/interminds/Desktop/cls11/'
    for clu, img_list in group.items():
        sp = save_path + str(clu) + '/'
        os.makedirs(sp, exist_ok=True)
        for imgs in img_list:
            shutil.copy(imgs, sp)

