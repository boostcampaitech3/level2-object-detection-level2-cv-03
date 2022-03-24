import multiprocessing
import os
from importlib import import_module

import torch
import pandas as pd

from dataset import CustomTestDataset
from dataset.transform import *
from utils import *
# import ttach as tta

# object detection
from pycocotools.coco import COCO

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def load_model(saved_model, num_classes, device, config):
    model_cls = getattr(import_module("model"), config.model.name)
    model = model_cls(
        num_classes=num_classes
    )
    if config.score == 'f1': # f1
        target_name = 'best_f1'
        model_path = os.path.join(saved_model, 'best_f1.pth')
    elif config.score == 'acc': # acc
        target_name = 'best_acc'
        model_path = os.path.join(saved_model, 'best_acc.pth')
    elif config.score == "loss": # loss
        target_name = 'best_loss'
        model_path = os.path.join(saved_model, 'best_loss.pth')
    else: # epoch
        target_name = 'epoch' + str(config.target_epoch)
        statedict_name = 'epoch' + str(config.target_epoch) + '.pth'
        model_path = os.path.join(saved_model, statedict_name)
    print(model_path)
    model.to(device)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, target_name


class Inferencer:
    def __init__(self, annotation, data_dir, model_dir, save_dir, score_threshold):
        self.annotation = annotation
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.score_threshold = score_threshold
        makedirs(self.save_dir)
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def inference(self, config):
        
        '''
        # -- transform
        transform_module = getattr(import_module("dataset"), config.augmentation.name)
        test_transform = transform_module(augment=False, **config.augmentation.args)

        # -- TTA transform
        if config.TTA.flag == True:
            try: 
                if config.TTA.transform == "roate":
                    TTA_list = [
                        tta.Rotate([0, 90, 270, 180]),
                    ]
                else:
                    TTA_list = [
                        tta.Scale([1, 1.03, 0.97]),
                        tta.Multiply([0.90, 1, 1.1]),
                        tta.Resize([(384, 512), (512, 512), (384, 384)]),
                    ]
            except:
                TTA_list = [
                    tta.Scale([1, 1.03, 0.97]),
                    tta.Multiply([0.90, 1, 1.1]),
                    tta.Resize([(384, 512), (512, 512), (384, 384)]),
                ]
            TTA_transform = tta.Compose(TTA_list)'''

        model, target_name = load_model(self.model_dir, 11, self.device, config)
        
        '''if config.TTA.flag == True:
            print("TTA is applied...")
            model = tta.ClassificationTTAWrapper(model, TTA_transform)'''
        
        # model.to(self.device)
        model.eval()
        test_dataset = CustomTestDataset(self.annotation, self.data_dir)
        test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            num_workers=multiprocessing.cpu_count()//2,
            pin_memory=self.use_cuda,
            **config.data_loader
        )

        print("Calculating inference results..")

        # inference_fn
        outputs = [] 
        for images in tqdm(test_data_loader):
            # gpu 계산을 위해 image.to(device)
            images = list(image.to(self.device) for image in images)
            output = model(config.flag, images)
            for out in output:
                outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})

        prediction_strings = []
        file_names = []
        coco = COCO(self.annotation)

        # submission 파일 생성
        for i, output in enumerate(outputs):
            prediction_string = ''
            image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
            for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                if score > self.score_threshold: 
                    # label[1~10] -> label[0~9]
                    prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                        box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
            prediction_strings.append(prediction_string)
            file_names.append(image_info['file_name'])
        submission = pd.DataFrame()
        submission['PredictionString'] = prediction_strings
        submission['image_id'] = file_names
        output_name = self.model_dir.split('/')[-1] + '_' + target_name + '.csv'
        submission.to_csv(os.path.join(self.save_dir, output_name), index=None)
        print(submission.head())


        '''
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(self.device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())
        model_name = self.model_dir.split('/')[-1]
        output_name = model_name + '_' + target_name + '.csv'
        if config.TTA.flag == True:
            try:
                output_name = "TTA_only_" + config.TTA.transform + "_" + output_name
            except:
                output_name = "TTA_combination" + output_name

        info = pd.read_csv(self.csv_path)

        info['ans'] = preds
        info.to_csv(os.path.join(self.save_dir, output_name), index=False)'''