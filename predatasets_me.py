from io import RawIOBase
import os
import shutil
import random
import copy
import glob
from tqdm import tqdm
import parmap
import multiprocessing
import numpy as np
import time
import albumentations as A
import cv2
from clustering import Clustering
from tqdm import tqdm


# class
class MakePreDataset:

    def __init__(self, input_path, output_path, label_list, aug, aug_num, false_ratio):
        self.label_list = label_list  # Dataset 만들고자 하는 label list
        self.input_path = input_path
        self.total_label_list = os.listdir(self.input_path)
        self.output_path = output_path
        self.num_cores = multiprocessing.cpu_count()
        self.aug = aug
        if self.aug == True:
            self.aug_num = aug_num
        self.false_ratio = false_ratio


    def augmentation(self, image_path, label):
        
        for path in image_path:
            image = cv2.imread(path)
            transform = A.Compose([
            A.RandomResizedCrop(height=300, width=300, scale=(0.9, 1.1), ratio = (0.9, 1.1), p = 1),
                ])
            augmented_image =transform(image=image)['image']
            num = random.randint(0,99999)
            cv2.imwrite(f'{self.output_path}/{label}/True/aug{num}_{label}.jpg', augmented_image)
        # return augmented_image

    def make_true(self, label):
        label_path = os.path.join(self.input_path, label, '0')
        label_pics = os.listdir(label_path)
        len_true_data = len(label_pics)

        true_folder_path = os.path.join(self.output_path, label, 'True')
        if os.path.exists(true_folder_path):
            print(label, '---> true 폴더 존재, 폴더 삭제 중')
            shutil.rmtree(true_folder_path)

        for pic in label_pics:
            os.makedirs(true_folder_path, exist_ok=True)
            file_name = f'{label}_'+pic
            shutil.copy(os.path.join(self.input_path, label, '0',pic), os.path.join(true_folder_path, file_name))

        if self.aug == True:
            true_data_target_count = self.aug_num - int(len_true_data)
            # num = 0
            print('추가되어야 하는 이미지 수 : ', true_data_target_count)

            true_image_path_list = glob.glob(os.path.join(true_folder_path, '*.jpg'))       

            random_image_path = random.choices(true_image_path_list, k= true_data_target_count)
            data = [x for x in random_image_path]
            splited_data = np.array_split(data, self.num_cores)
            splited_data = [x.tolist() for x in splited_data]

            parmap.map(self.augmentation, splited_data, label, pm_pbar=True, pm_processes = self.num_cores)


    def get_versions(self, pog_name):
        label_paths = os.path.join(self.input_path, pog_name)
        all_version_list = os.listdir(label_paths) #모든 ver
        latest_version = all_version_list[-1] #가장 마지막 ver
        return all_version_list, latest_version

    # False 라벨 리스트 가져오기 - 같은 상품 다른 사이즈 데이터 제외 
    def calc_false_label_list(self, true_label):
        # all_dir = os.listdir(self.input_path)
        
        labels_copy = copy.copy(self.total_label_list)
        labels_copy.remove(true_label)
        
        ori_true_label = '_'.join(true_label.split('_')[:-1])
        len_ori_true_label = len(ori_true_label)

        a = [i.replace('_can','').replace('_tspn','') for i in labels_copy]

        idx_list = []
        for idx, item in enumerate(a):
            if item[:len_ori_true_label] == ori_true_label:
                idx_list.append(idx)

        final_label_list = copy.copy(labels_copy)
        for i in idx_list:
            final_label_list.remove(labels_copy[i])

        return final_label_list

    def make_false(self, label):
        
        false_path = os.path.join(self.output_path, label, 'False')

        if os.path.exists(false_path):
            print(label, '---> False 폴더 존재, 폴더 삭제 중')
            shutil.rmtree(false_path)
        os.makedirs(false_path, exist_ok=True)

        false_label_list = self.calc_false_label_list(true_label = label)
        # print('false label list : ', false_label_list)
        
        # clust = Clustering('resnet26')

        false_folder_count = len(false_label_list)
        # false data 총 개수
        if self.aug == True:
            target_count = int(self.aug_num * self.false_ratio)
        else:
            target_count = int(len(glob.glob(os.path.join(self.input_path, label, '0')+'/*.jpg')) * self.false_ratio)  # 랜덤으로 가져올 label folder 종류 개수

        for i in tqdm(false_label_list, desc = 'Make False Data'):
            path = glob.glob(self.input_path + '/' + i + '/0' + '/*.jpg')
            count2 = int(target_count/false_folder_count)
            
            if len(path) < count2:
                count2 = len(path)
            
            # group = clust.clustering(images = path, k = count2)
            
            # for clu, img_list in tqdm(group.items(), desc = 'False image Clustering and Copying'):
            #     # sp = save_path + str(clu) + '/'
            #     # os.makedirs(sp, exist_ok=True)
            #     for imgs in random.sample(img_list, 1):
            #         # shutil.copy(imgs, sp)
            #         shutil.copy(imgs, os.path.join(false_path, f'{i}_' + f'c{clu}_'+imgs.split('/')[-1]))


            for file_path in random.sample(path, count2):
                # os.makedirs(false_path, exist_ok=True)
                shutil.copy(file_path, os.path.join(false_path, f'{i}_' + file_path.split('/')[-1]))

    def make_datasets(self, label_list):
        
        for label in label_list:
            print(f'{label} -> True 데이터 생성 중...')
            self.make_true(label = label)
            self.make_false(label = label)
            print(f'{label} -> False 데이터 생성 중...')
            print('-' * 50)
            

    def run(self): # 모든 label에서 랜덤으로 생성
        start = time.time()
        
        data = [x for x in self.label_list]

        print(data)
        splited_data = np.array_split(data, self.num_cores)
        splited_data = [x.tolist() for x in splited_data]
        print(splited_data)

        parmap.map(self.make_datasets, splited_data, pm_pbar=True, pm_processes = self.num_cores)
        end = time.time()

        print(f"{end - start:.3f} sec")


def get_labels(path):  # label.txt로 부터 list 가져오기
    with open(path, 'r') as file:
        labels = file.readlines()
        labels = list(map(lambda x : x.strip(), labels))
        return labels

if __name__ == '__main__':

    # label_list = ['big_wave_golden_ale_473','blanche_1866_500_can_tspn','cass_fresh_500','cass_light_500','fil_good_500','filite_fresh_500','gompyo_summer_ale_500_can_tspn','reeper_b_ipa_500_can_tspn']
    # label_list = ['terra_500', 'tsingtao_500', 'tsingtao_non_330']
    label_list = ['cass_fresh_500', 'cass_fresh_355', 'cass_light_500']

    # md = MakePreDataset(label_list = label_list, input_path = '/data/1_sr_rnd/make_false_data', output_path = './result')
    md = MakePreDataset(label_list = label_list, input_path = '/data/1_sr_rnd/0_sr_rnd_datasets/seed_data', output_path = './result')
    # md.make_false_from_total_label()
    # aa = md.calc_false_label_list(true_label = 'cass_fresh_355')
    # print(aa)

    # md.make_true(label = 'cass_fresh_500')
    # md.make_false(label = 'cass_fresh_500')
    md.run()

    # print(md.get_versions('tsingtao_non_330'))
