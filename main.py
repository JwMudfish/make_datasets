from fastapi import FastAPI
from pydantic import BaseModel
from predatasets_me import MakePreDataset

class Item(BaseModel):
    label_list : list = None
    true_aug : bool = None
    true_aug_num : int = None
    false_ratio : float = None
    
app = FastAPI()
# uvicorn main:app --reload
@app.post('/make_datasets/')
async def make_datasets(item: Item):
        md = MakePreDataset(label_list = item.label_list,
                            input_path = '/data/1_sr_rnd/0_sr_rnd_datasets/seed_data', 
                            output_path = '/data/1_sr_rnd/0_sr_rnd_datasets/train_data',
                            aug = item.true_aug,
                            aug_num = item.true_aug_num,
                            false_ratio = item.false_ratio)
        md.run()
        return 'done'