import requests
import json

# 2. 요청 보낼 url 주소
url = "http://192.168.0.92:8000/make_datasets/"

# 3. 같이 보낼 데이터 작성
# data = {"name": "new_challenge", "description":"test1", "price":2020, "tax":2021}
data = {'label_list' : ['jinro_soju_pet_640'],
        'true_aug' : True,
        'true_aug_num' : 1500,
        'false_ratio' : 2.5}

# 4. post로 API서버에 요청보내기
res = requests.post(url, data=json.dumps(data))

# 5. 결과 확인하기
print(res.text)