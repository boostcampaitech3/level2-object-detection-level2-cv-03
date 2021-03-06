# ๐ Object Detection for Recycling Trash
<br/>

## ๐จโ๐พ Team
* Level 2 CV Team 03 - ๋น๋จ์ฝ์ธ
* ํ ๊ตฌ์ฑ์ : ๊น๋๊ทผ, ๋ฐ์ ํ, ๊ฐ๋ฉด๊ตฌ, ์ ์ฌ์ฑ, ํํ์ง

<br/>

## ๐ Main Subject
๋๋ ์์ฐ, ๋๋ ์๋น์ ์๋์์๋ ํ์ฐ์ ์ผ๋ก โ์ฐ๋ ๊ธฐ ์ฒ๋ฆฌโ๋ฌธ์ ๊ฐ ๋ฐ์ํฉ๋๋ค. ๋ถ๋ฆฌ ์๊ฑฐ๋ ์ด๋ฌํ ํ๊ฒฝ ๋ถ๋ด์ ์ค์ผ ์ ์๋ ๋ํ์ ์ธ ๋ฐฉ๋ฒ์ด๋ฉฐ, ์ฌ๋ฐ๋ฅธ ๋ฐฉ์์ผ๋ก ๋ถ๋ฆฌ์๊ฑฐ๋ฅผ ์ํํด์ผ ํฉ๋๋ค.

ํด๋น ํ๋ก์ ํธ์์๋ ์ฌ์ง์์ ์ฐ๋ ๊ธฐ๋ฅผ detectionํ๋ ๋ชจ๋ธ์ ๋ง๋ค์ด ๋ถ๋ฆฌ์๊ฑฐ๋ฅผ ์งํํ๊ณ ์ ํฉ๋๋ค. ํนํ 10๊ฐ์ง๋ก ๋๋๋ ์ฐ๋ ๊ธฐ ์ข๋ฅ์ ์์น๋ฅผ ํ์ํ๊ธฐ ์ํ ๋ชจ๋ธ์ ๋ง๋๋ ๊ฒ์ ์ง์คํฉ๋๋ค.  
<br/>

## ๐ป Development Environment
**๊ฐ๋ฐ ์ธ์ด** : PYTHON (IDE: VSCODE, JUPYTER NOTEBOOK)

**์๋ฒ**: AI STAGES (GPU: NVIDIA TESLA V100)

**ํ์ Tool** : git, notion, [wandb](https://wandb.ai/cv-3-bitcoin), [google spreadsheet](https://docs.google.com/spreadsheets/d/1l-sIS-KdCHQUgI3Y1CrjsbkTQA0p5vqaErDRQ_eifCo/edit#gid=0), slack

**Library** : mmdetection

<br/>

## ๐ฟ Project Summary
  - **Data Augmentation**
    - MultiScale, Flip, Blur, Rotate, Brightness, HueSaturation, GaussianNoise, sharpen
  - **TTA**
    - Inference(Test) ๊ณผ์ ์์ Augmentation ์ ์ ์ฉํ ๋ค ์์ธก์ ํ๋ฅ ์ ํ๊ท (๋๋ ๋ค๋ฅธ ๋ฐฉ๋ฒ)์ ํตํด ๋์ถํ๋ ๊ธฐ๋ฒ
    - Multiscale โ 0.75, 1.0, 1.25, 1.5, 1.75, 2์ ratio๋ฅผ ์ฌ์ฉ.
    - Flip โ Horizontal & Vertical
  - **Ensemble**
    - nms, soft-nms, wbf

### Dataset
  - ๋ฐ์ดํฐ๋ ์ฌํ์ฉ ์ฐ๋ ๊ธฐ๊ฐ ์ดฌ์๋ **.jpg ํ์์ ์ด๋ฏธ์ง** ์ **bbox์ ์์น** ๋ฐ ์ข๋ฅ๋ฅผ ๋ช์ํ **.json ํ์ผ**๋ก ์ด๋ฃจ์ด์ ธ ์์ผ๋ฉฐ ๊ฐ๊ฐ train, test๋ก ๊ตฌ๋ถ๋์ด ์์  
    ![image](https://user-images.githubusercontent.com/78528903/168411862-7dd2b68a-c3f9-4ada-8a73-da038e735853.png)
  - **๋ฒ์ฃผ** : ๋ฐฐ๊ฒฝ, ์ผ๋ฐ์ฐ๋ ๊ธฐ, ์ข์ด, ์ข์ดํฉ, ๊ธ์, ์ ๋ฆฌ, ํ๋ผ์คํฑ, ์คํฐ๋กํผ, ํ๋ผ์คํฑ ๊ฐ๋ฐฉ, ๋ฐฐํฐ๋ฆฌ, ์๋ฅ ์ด 11๊ฐ์ง
### Metrics
  - **mAP50(Mean Average Precision)**
    - Object Detection์์ ์ฌ์ฉํ๋ ๋ํ์ ์ธ ์ฑ๋ฅ ์ธก์  ๋ฐฉ๋ฒ
    - Ground Truth ๋ฐ์ค์ Prediction ๋ฐ์ค๊ฐ IoU(Intersection Over Union, Detector์ ์ ํ๋๋ฅผ ํ๊ฐํ๋ ์งํ)๊ฐ 50์ด ๋๋ ์์ธก์ ๋ํด True๋ผ๊ณ  ํ๋จํฉ๋๋ค.
    
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
