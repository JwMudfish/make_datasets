import cv2
import torch
import numpy as np
from logic.models.superpoint import SuperPoint


class Matching(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.004,
                'max_keypoints': -1
            }
        }
        self.superpoint = SuperPoint(config).to('cuda').eval()
        self.show_keypoints = True

    @staticmethod
    def frame2tensor(frame, device):
        return torch.from_numpy(frame / 255.).float()[None, None].to(device)

    @staticmethod
    def white_balance(img):
        result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        return result

    def pre_processing(self, img):
        img = self.white_balance(img)  # rgb
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        frame_tensor = self.frame2tensor(img, 'cuda')
        return frame_tensor

    @torch.no_grad()
    def get_max_keypoints(self, det_res):
        max_key = {}
        for cluster, images in det_res.items():
            max_key[cluster] = []
            for img in images:
                frame_tensor = self.pre_processing(img)
                sp = self.superpoint({'image': frame_tensor})

                kpts0 = sp['keypoints'][0].cpu().numpy()
                # scores = sp['scores'][0].cpu().numpy()

                max_key[cluster].append(len(kpts0))

        max_indexes = {cluster: keypoint_vals.index(max(keypoint_vals)) for cluster, keypoint_vals in max_key.items()}
        max_images = {c: i[max_indexes[c]] for c, i in det_res.items()}
        return max_images

