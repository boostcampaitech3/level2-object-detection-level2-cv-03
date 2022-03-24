# Baseline by torchvision
## Dependecy 
터미널 창에서 다음의 코드를 입력합니다.
>```
>pip insatll -r requirements.txt
>```
(실험을 진행하면서 수정한 부분도 있긴해서 requirements 수정도 필요하긴 할 것 같습니다만... 안되는거 있으시면 알려주세요!)

## Train
먼저 train_config.yaml 파일에서 원하는 실험을 구성한 후 다음의 코드를 입력합니다.
>```
>python train.py -c train_config.yaml
>```

train을 진행하면서 에폭 단위별로, 최저 loss를 갱신하며 각각 pth 파일을 model_weights 폴더에 저장합니다.

## Inference
마찬가지로 inference_config.yaml 파일을 구성한 뒤 다음을 실행합니다.
>```
>python inference.py -c inference_config.yaml
>```
train 단계에서 얻은 pth 파일을 지정하여 inference를 진행합니다.

