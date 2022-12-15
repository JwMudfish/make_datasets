import os
import cv2
import math
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

from logic.clustering import Clustering
from logic.detection import Detection
from logic.matching import Matching

from skimage.color import rgb2lab
# from error_code.error_code import *


class AutoFeature:
    def __init__(self):
        # --- setup ---
        self.cls = Clustering('resnet26')
        self.det = Detection()
        self.mat = Matching()

        self.keys = ['keypoints', 'scores', 'descriptors']
        self.k = 100  # default k

        self.root_path = './save/'

    def image_load(self, images):
        # --- 0. image load ---
        # assuming the request was sent as RGB format
        images = {idx: self.det.base64_to_array(e_img) for idx, e_img in enumerate(images)}
        return images

    def save_images(self, pog_name, version, mat_res, plt_cls=None):
        # --- 4. save result ---
        # --- save clustering result ---
        save_path = os.path.join(self.root_path, 'img', str(version), pog_name)
        os.makedirs(f'{save_path}/res/', exist_ok=True)
        if plt_cls is not None:
            plt_cls.savefig(f'{save_path}/res/{pog_name}_cluster.png')
            plt.clf()

        # --- save image ---
        for cluster, image in mat_res.items():
            cv2.imwrite(f'{save_path}/{pog_name}_{cluster}.jpg', image[:, :, ::-1])

            # --- save clustering image ---
            plt.subplot(math.ceil(self.k/5), 5, cluster + 1)
            plt.imshow(image)
            plt.axis('off')
        plt.savefig(f'{save_path}/res/{pog_name}_sample.png')
        plt.clf()

    @staticmethod
    def histogram_equalization(rgb_img):
        ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)
        return equalized_img

    def get_principal_color(self, img):
        img = self.histogram_equalization(img)
        data = np.float32(np.reshape(img, (-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, labels, centers = cv2.kmeans(data, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        principal_color = centers[0].astype(np.int32)
        lab = rgb2lab(np.uint8(np.asarray([principal_color])))
        return lab

    def make_reference_pickle(self, images, pog_name, version):
        ref = [self.make_ref(i) for _, i in images.items()]
        save_path = f'{self.root_path}/feature/{version}/'
        os.makedirs(save_path, exist_ok=True)
        with open(save_path + f'{pog_name}.pickle', 'wb') as f:
            pickle.dump(ref, f)

    def make_ref(self, img):
        frame_tensor = self.mat.pre_processing(img)  # rgb -> rgb2gray -> tensor
        last_data = self.mat.superpoint({'image': frame_tensor})
        last_data = {k + '0': last_data[k] for k in self.keys}
        last_data['image0'] = frame_tensor
        last_data['principal_color'] = self.get_principal_color(img)
        return last_data

    def run(self, images, pog_name, version):
        try:
            images = self.image_load(images)  # maybe..rgb..?

            if len(images) > self.k:
                # 1. clustering
                groups, plt_cls = self.cls.clustering(images, self.k)

                # 2. detection
                det_res = self.det.detection(groups)

                # 3. matchingd
                mat_res = self.mat.get_max_keypoints(det_res)
            else:
                mat_res = images
                plt_cls = None

            # 4. save images
            self.save_images(pog_name, version, mat_res, plt_cls)

            # 5. make feature
            self.make_reference_pickle(mat_res, pog_name, version)

            return f'{pog_name} feature pickle file saving complete!'

        # except MakeDataError as e:
        #     raise MakeDataError(code=e.code, msg=e.message)

        except Exception as e:
            print(e)

if __name__ == '__main__':
    path = '/data/1_sr_rnd/make_false_data/cass_fresh_500/0/'
    files = glob.glob(path + '*.jpg')

    af = AutoFeature()
    af.run(images = files, name = 'cass_fresh_500', version = '0')