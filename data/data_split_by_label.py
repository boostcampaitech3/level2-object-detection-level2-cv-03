#%% Import Libraries
import os
import json

from pycocotools.coco import COCO
from copy import deepcopy
from random import sample

#%% Load train and validation data
data_dir = '../../dataset' # data_dir 경로

annot_train = '../../dataset/cv_train_1.json' # Multilabel K-Fold 방식으로 분리된 train set의 annotation
annot_valid = '../../dataset/cv_val_1.json' # Multilabel K-Fold 방식으로 분리된 validation set의 annotation

with open(annot_train) as f:
    data_train = json.load(f)
    
with open(annot_valid) as f:
    data_valid = json.load(f)
    
data_train_major = deepcopy(data_train); data_train_minor = deepcopy(data_train)
data_valid_major = deepcopy(data_valid); data_valid_minor = deepcopy(data_valid)

#%% Divide each dataset by class label

## Set major and minor categories
major_category = [1, 7, 0, 5] # Paper, Plastic bag, General trash, Plastic
minor_category = [x for x in range(1, 10) if x not in major_category] # Styrofoam, Glass, Metal, Paper pack, Clothing, Battery
print(f'Major categories: {major_category}')
print(f'Minor categories: {minor_category}')

# Remove unnecessary annotations
data_train_major['annotations'] = [x for x in data_train['annotations'] if x['category_id'] in major_category]
data_train_minor['annotations'] = [x for x in data_train['annotations'] if x['category_id'] in minor_category]

data_valid_major['annotations'] = [x for x in data_valid['annotations'] if x['category_id'] in major_category]
data_valid_minor['annotations'] = [x for x in data_valid['annotations'] if x['category_id'] in minor_category]

# Remove unnecessary category id
data_train_major['categories'] = [x for x in data_train['categories'] if x['id'] in major_category]
data_train_minor['categories'] = [x for x in data_train['categories'] if x['id'] in minor_category]

data_valid_major['categories'] = [x for x in data_valid['categories'] if x['id'] in major_category]
data_valid_minor['categories'] = [x for x in data_valid['categories'] if x['id'] in minor_category]

# Remove unnecessary image infomations
img_w_ann_train_major = sorted(list(set([x['image_id'] for x in data_train_major['annotations']])))
img_w_ann_train_minor = sorted(list(set([x['image_id'] for x in data_train_minor['annotations']])))
img_w_ann_valid_major = sorted(list(set([x['image_id'] for x in data_valid_major['annotations']])))
img_w_ann_valid_minor = sorted(list(set([x['image_id'] for x in data_valid_minor['annotations']])))

data_train_major['images'] = [x for x in data_train['images'] if x['id'] in img_w_ann_train_major]
data_train_minor['images'] = [x for x in data_train['images'] if x['id'] in img_w_ann_train_minor]
data_valid_major['images'] = [x for x in data_valid['images'] if x['id'] in img_w_ann_valid_major]
data_valid_minor['images'] = [x for x in data_valid['images'] if x['id'] in img_w_ann_valid_minor]

#%% Save new data
def save_json(data: dict, file_nm: str, dir_path=data_dir):
    with open(os.path.join(data_dir, file_nm), 'w') as outfile:
        json.dump(data, outfile)
              

data_list = [data_train_major,
             data_train_minor, 
             data_valid_major, 
             data_valid_minor
            ]
file_nm_list = ['cv_train_1_major.json',
                'cv_train_1_minor.json',
                'cv_val_1_major.json',
                'cv_val_1_minor.json',
               ]

for data, file_nm in zip(data_list, file_nm_list):
    save_json(data, file_nm)
    
#%% Test json result
with open(os.path.join(data_dir, file_nm_list[0])) as f:
    data_test = json.load(f)
    
print(*sample(data_test['annotations'], 5), sep='\n')