from cProfile import label
from distutils.file_util import copy_file
import os
import glob
import shutil
import random
from pprint import pprint
from tqdm import tqdm
import copy
import parmap
import time

def get_labels(path):  # label.txt로 부터 list 가져오기
    with open(path, 'r') as file:
        labels = file.readlines()
        labels = list(map(lambda x : x.strip(), labels))
        return labels

# True False 디렉토리 만들기
class MakePreDataset:
    def __init__(self, input_path, output_path, label_list):
        self.label_list = label_list
        self.input_path = input_path
        self.output_path = output_path
        if not self.check_initialze():
            print("check label_list.")
            # exit()

    def check_initialze(self):
        if len(self.label_list) == 0:
            return False
        else:
            return True

    def get_versions(self, pog_name):
        label_paths = os.path.join(self.input_path, pog_name)
        all_version_list = os.listdir(label_paths) #모든 ver
        latest_version = all_version_list[-1] #가장 마지막 ver
        return all_version_list, latest_version

    def random_List(self, list_size, max_num):
        result = []
        if max_num < list_size:
            list_size = max_num
        for v in range(int(90000000000)):
            result.append(random.randint(0, max_num-1))
            result = list(set(result))
            if len(result) == list_size:
                break
        return result


    def make_dataset(self, pog_name, version_name, output_path, delimiter, sampling_num = -1):

        if sampling_num == -1:
            version_path = os.path.join(self.input_path, pog_name, version_name)
            output_path = os.path.join(self.output_path, pog_name, delimiter)

            if os.path.exists(os.path.join(self.output_path, pog_name)):
                # st.write(f"{pog_name} 폴더 존재함. 삭제 후 다시 만드는 중")
                shutil.rmtree(os.path.join(self.output_path, pog_name))
                shutil.copytree(version_path, output_path)
            else:
                shutil.copytree(version_path, output_path)
                # st.write(f"{pog_name} 데이터셋 만드는 중")
        else:
            version_path = os.path.join(self.input_path, pog_name, version_name)
            output_path = os.path.join(self.output_path, pog_name, delimiter)
            versions_file_list = glob.glob1(version_path, "*.jpg")
            copy_index_list = self.random_List(sampling_num, len(versions_file_list))
            copy_file_list = []
            for index in copy_index_list:
                copy_file_list.append(versions_file_list[index])
            for file_name in copy_file_list:
                src = os.path.join(version_path, file_name)
                dst = os.path.join(output_path, file_name)
                shutil.copy(src, dst)


    def make_dataset_for_full_path_list(self, pog_label, full_path_list, output_path, delimiter, sampling_num):
        copy_index_list = self.random_List(sampling_num, len(full_path_list))
        copy_file_list = []
        for index in copy_index_list:
            copy_file_list.append(full_path_list[index])
        for full_path in copy_file_list:
            #label = full_path.split('/')[2]
            os.makedirs(os.path.join(output_path, pog_label, delimiter), exist_ok=True)
            file_name = os.path.basename(full_path)
            src = os.path.join(full_path)
            dst = os.path.join(output_path, pog_label, delimiter, file_name)
            shutil.copy(src, dst)


    def make_true_data(self):
        # 1. 최신 버전은 전부 가져와야 함
        # 2. 나머지 버전은 균등하게 가져와야 함, 어떻게?
        # : 나머지 각 폴더에서 균등하게 가져와야 할 개수
        #  = 최신버전의 파일 개수 / 나버지 버전의 개수
        print('True Data 가져오는 중')
        for label in tqdm(self.label_list):
            all_version_list, latest_version = self.get_versions(label)
            # 1. 최신 버전 가져오기
            self.make_dataset(label, latest_version, self.output_path, "True", -1)
            # 2. 나머지 버전은 균등하게 가져와야 함
            if len(all_version_list) > 1:
                left_version_list = all_version_list[:-1]
            else:
                left_version_list = all_version_list[0]

            latest_file_num = len(glob.glob1(os.path.join(self.input_path, label, latest_version), "*.jpg"))
            sampling_num = latest_file_num / len(left_version_list)

            for version_name in left_version_list:
                self.make_dataset(label, version_name, self.output_path, "True", sampling_num)


    def make_image_file_list(self, pog_name):
        image_file_list = []
        version_list = os.listdir(os.path.join(self.input_path, pog_name))

        for version_name in version_list:
            path = os.path.join(self.input_path, pog_name, version_name)
            file_list = glob.glob1(path, "*.jpg")
            file_list = [os.path.join(path, val) for val in file_list]
            image_file_list += file_list

        return image_file_list

    # True의 총 사진 개수만큼 가져와야 하는데....
    # 나머지 pog 폴더에서 그리고 그 각 pog 폴더의 version은 균등하게 하고...
    # 균등하게(랜덤하게) 가져와야 한다.


    def make_false_data(self):
        self.make_true_data()
        for label in self.label_list:
            all_label_list = os.listdir(self.input_path)
            label_list_dummy = copy.deepcopy(all_label_list)
            label_list_dummy.remove(label)
            true_num = len(glob.glob1(os.path.join(self.output_path, label, "True"), "*.jpg"))
            left_pog_num = len(os.listdir(self.input_path)) - 1
            file_num_for_folder = int(true_num / left_pog_num)
            # file_num_for_folder = true_num
            for left_label in label_list_dummy:
                full_path_list = self.make_image_file_list(left_label)
                self.make_dataset_for_full_path_list(label, full_path_list, self.output_path, "False", file_num_for_folder)
            # st.success(f'"{label}" Complete')


if __name__ == '__main__':
    # label_list = get_labels('./dataset.txt')
    start = time.time()
    label_list = ['kloud_500']
    # label_list = ['belgium_export_500',
    # 'belgium_pilsner_500',
    # 'belgium_weizen_500',
    # 'bernini_lemon_500',
    # 'bernini_strawberry_500',
    # 'big_wave_golden_ale_473',
    # 'blanc1664_330_can_tspn',
    # 'blanc1664_500',
    # 'blanche_1866_500_can_tspn',
    # 'blonde_1866_500_can_tspn']
    
    md = MakePreDataset(label_list = label_list, 
                        input_path = '/data/product101/total_datasets',
                        output_path = '/data/product101/train_datastes')
    # md.make_true_data()
    md.make_false_data()

    end = time.time()

    print(f"{end - start:.3f} sec")

