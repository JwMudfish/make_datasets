import cv2
import json
import base64
import urllib3
import requests
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Detection:
    def __init__(self):
        self.api_url = 'https://interminds-ai.com/v0/inference'
        self.api_key = 'pTsHkPdCyeFDolBRywQTcXurdbxBL7K2kstMoTHs'
        self.api_ids = 'book'

    @staticmethod
    def encode(img: np.ndarray) -> str:
        # img = cv2.resize(img, (224, 224))  # if product101(fridge & classification)
        # img = img[:, :, ::-1]  # bgr to rgb, already rgb format
        img = cv2.imencode('.jpg', img)[1].tobytes()  # .png or .jpg
        img = base64.b64encode(img).decode('utf-8')
        return img

    @staticmethod
    def base64_to_array(encoded_str):
        im_bytes = base64.b64decode(bytes(encoded_str, 'utf-8'))
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        return img

    @staticmethod
    def area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def request(self, payload):
        # request batch try
        for _ in range(3):
            try:
                response = requests.post(
                    url=self.api_url,
                    headers={
                        "X-API-KEY": self.api_key,
                        "USER-ID": self.api_ids
                    },
                    data=json.dumps(payload),
                    verify=False
                )
                boxes = response.json()['res']
                return boxes
            except:
                 continue

        # in fail case, request one by one.
        boxes = []
        for i in range(len(payload['images'])):
            small_payload = {
                'images': [payload['images'][i]],
                'device_type': 'fridge',  # fridge, cigar
                'model_type': 'detection',  # classification, detection
                'model_name': 'short_tall',  # 00888_c_test, JJ0001_c_0001, short_tall, cass, or [cass, terra], ...
            }
            try:
                response = requests.post(
                    url=self.api_url,
                    headers={
                        "X-API-KEY": self.api_key,
                        "USER-ID": self.api_ids
                    },
                    data=json.dumps(small_payload),
                    verify=False
                )
                boxes.append(response.json()['res'][0])
            except:
                 boxes.append([[0, 0, 0, 0]])
        return boxes


    @staticmethod
    def make_batch(data, batch_size):
        mini_batch = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return mini_batch

    def detection(self, groups):
        det_refine_res = {}
        for cluster, images in groups.items():
            boxes = []

            mini_batch = self.make_batch(images, 50)
            for batch in mini_batch:
                encode_images = [self.encode(img) for img in batch]
                payload = {
                    'images': encode_images,
                    'device_type': 'fridge',
                    'model_type': 'detection',
                    'model_name': 'short_tall',
                }
                boxes.append(self.request(payload))

            boxes = [b for box in boxes for b in box]

            # only images with one box and the top 25% area passed!
            refine_images_num = {n: self.area(boxes[n][0][:4]) for n, b in enumerate(boxes) if len(b) == 1}

            if len(refine_images_num) == 0:
                det_refine_res[cluster] = images
            else:
                percent = int(len(refine_images_num) * 0.25)
                refine_images_num = [k for k, _ in sorted(refine_images_num.items(), key=lambda x: x[1], reverse=True)[
                                                   :percent if percent > 0 else 1]]
                refine_images = [images[re_num] for re_num in refine_images_num]
                det_refine_res[cluster] = refine_images
        return det_refine_res
