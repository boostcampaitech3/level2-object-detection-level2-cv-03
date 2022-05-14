# 🌏 Object Detection for Recycling Trash
<br/>

## 👨‍🌾 Team
* Level 2 CV Team 03 - 비뜨코인
* 팀 구성원 : 김대근, 박선혁, 강면구, 정재욱, 한현진

<br/>

## 🎇 Main Subject
대량 생산, 대량 소비의 시대에서는 필연적으로 “쓰레기 처리”문제가 발생합니다. 분리 수거는 이러한 환경 부담을 줄일 수 있는 대표적인 방법이며, 올바른 방식으로 분리수거를 수행해야 합니다.

해당 프로젝트에서는 사진에서 쓰레기를 detection하는 모델을 만들어 분리수거를 진행하고자 합니다. 특히 10가지로 나뉘는 쓰레기 종류와 위치를 파악하기 위한 모델을 만드는 것에 집중합니다.  
<br/>

## 💻 Development Environment
**개발 언어** : PYTHON (IDE: VSCODE, JUPYTER NOTEBOOK)

**서버**: AI STAGES (GPU: NVIDIA TESLA V100)

**협업 Tool** : git, notion, [wandb](https://wandb.ai/cv-3-bitcoin), [google spreadsheet](https://docs.google.com/spreadsheets/d/1l-sIS-KdCHQUgI3Y1CrjsbkTQA0p5vqaErDRQ_eifCo/edit#gid=0), slack

**Library** : mmdetection

<br/>

## 🌿 Project Summary
  - **Data Augmentation**
    - MultiScale, Flip, Blur, Rotate, Brightness, HueSaturation, GaussianNoise, sharpen
  - **TTA**
    - Inference(Test) 과정에서 Augmentation 을 적용한 뒤 예측의 확률을 평균(또는 다른 방법)을 통해 도출하는 기법
    - Multiscale → 0.75, 1.0, 1.25, 1.5, 1.75, 2의 ratio를 사용.
    - Flip → Horizontal & Vertical
  - **Ensemble**
    - nms, soft-nms, wbf

### Dataset
  - 데이터는 재활용 쓰레기가 촬영된 **.jpg 형식의 이미지** 와 **bbox의 위치** 및 종류를 명시한 **.json 파일**로 이루어져 있으며 각각 train, test로 구분되어 있음  
    ![image](https://user-images.githubusercontent.com/78528903/168411862-7dd2b68a-c3f9-4ada-8a73-da038e735853.png)
  - **범주** : 배경, 일반쓰레기, 종이, 종이팩, 금속, 유리, 플라스틱, 스티로폼, 플라스틱 가방, 배터리, 의류 총 11가지
### Metrics
  - **mAP50(Mean Average Precision)**
    - Object Detection에서 사용하는 대표적인 성능 측정 방법
    - Ground Truth 박스와 Prediction 박스간 IoU(Intersection Over Union, Detector의 정확도를 평가하는 지표)가 50이 넘는 예측에 대해 True라고 판단합니다.
    
        ![image](https://user-images.githubusercontent.com/78528903/168411911-3869c61a-bb2a-465d-9084-d43bd507ea5a.png)

### Model

|Model|library|LB Score@public|LB Score@private|
|:---:|:---:|---:|---:|
|UniverseNet101|mmdetection|0.5962|0.5750|
|Yolo_v5||0.5331|0.5149|
|Swin-T|mmdetection|0.4782|0.4615|
|Swin-S|mmdetection|0.4713|0.4601|
|Swin-L|mmdetection|0.5481|0.5393|

## [Wrap Up Report](https://www.notion.so/Wrap-Up-1-dafff80131d34f798be2bef2c3f09585)
