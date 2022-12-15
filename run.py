from flask import Flask, request
from predatasets_me import MakePreDataset

app = Flask (__name__)
 
@app.route('/predatasets', methods = ['POST'])
def make_datasets():
    return "POST 요청 내용은{}".format(json.loads(request.get_data()))
    # print(a)
    # return a.to_dict()



if __name__ == "__main__":
    app.run(host = '192.168.0.153', port = 1818)
