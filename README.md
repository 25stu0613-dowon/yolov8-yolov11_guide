# yolov8-yolov11_guide


````markdown
# YOLOv8 / YOLOv11 정리 & 비교  
> 이 문서는 객체 탐지 모델인 YOLO 계열 중에서 YOLOv8과 YOLOv11을 중심으로 초보자가 깃허브에 정리할 수 있도록 만든 마크다운 템플릿입니다.

## 1. 프로젝트 개요  
이 리포지토리는 **YOLO 기반 객체탐지(Object Detection)** 모델을 공부하고 실습해 보면서 정리한 자료입니다.  
특히 최신 버전인 YOLOv11을 중심으로, 이전 버전인 YOLOv8과의 차이를 이해하고, 직접 실습 가능한 형태로 기록합니다.

목표 예시:  
- YOLOv8, YOLOv11의 구조와 특징 비교  
- 커스텀 데이터셋으로 학습해 보기  
- 실습 코드, 결과 시각화, 느낀 점 기록  
- 용어 정리 → 초보도 이해 가능하도록  

## 2. 버전 비교: YOLOv8 vs YOLOv11  
아래 표는 일반적인 비교 항목을 정리한 것입니다. (자료 출처 참고)  

| 항목 | YOLOv8 | YOLOv11 |
|------|--------|---------|
| 주요 특징 | 비교적 안정된 최신 버전. 사용자가 많은 자료 존재. :contentReference[oaicite:2]{index=2} | YOLOv8 대비 **더 적은 파라미터**, **더 빠른 추론 속도**, **높은 정확도**를 목표로 설계됨. :contentReference[oaicite:3]{index=3} |
| mAP / 정확도 예시 | 모델 크기나 종류에 따라 달라지나, 예시로 YOLOv8l 기준 mAP 약 52.9% (640px 입력) :contentReference[oaicite:4]{index=4} | 예시로 YOLOv11l 기준 mAP 약 53.4% (640px 입력) 및 파라미터 감소, FLOPs 감소. :contentReference[oaicite:5]{index=5} |
| 파라미터(Parameters) & FLOPs 예시 | 예: YOLOv8l → 파라미터 약 43.7M, FLOPs 약 165.2B :contentReference[oaicite:6]{index=6} | 예: YOLOv11l → 파라미터 약 25.3M, FLOPs 약 86.9B :contentReference[oaicite:7]{index=7} |
| 추론 속도 / CPU ONNX 예시 | YOLOv8n 기준 CPU ONNX 약 80.4ms (640px) :contentReference[oaicite:8]{index=8} | YOLOv11n 기준 CPU ONNX 약 56.1ms (640px) :contentReference[oaicite:9]{index=9} |
| 사용성 & 참고 자료 | 자료, 튜토리얼 많음 → 입문자에 유리 | 최신 버전으로 개선 많지만 사용자 자료가 YOLOv8보다는 조금 적을 수 있음 |
| 주의사항 | 충분히 검증된 버전 | 새로운 구조/최적화가 많아 커스텀 데이터에서 기대만큼 나오지 않을 가능성 있음 (데이터셋/하이퍼파라미터 중요) :contentReference[oaicite:10]{index=10} |

> 요약:  
> – 입문자라면 YOLOv8부터 시작해서 익숙해진 뒤 YOLOv11로 넘어가는 것도 좋은 전략입니다.  
> – 만약 하드웨어 자원이 괜찮고 최신 버전을 써보고 싶다면 YOLOv11을 바로 적용해보는 것도 좋습니다.

## 3. 용어 정리  
다음은 객체탐지나 YOLO 계열을 공부할 때 자주 나오는 용어들입니다.

| 용어 | 뜻 | 설명 |
|------|-----|------|
| 객체탐지(Object Detection) | 이미지나 영상에서 **여러 객체의 위치(바운딩 박스)와 클래스(무엇인지)를 동시에 인식**하는 기술 | 이미지 분류(Classification)와 달리 여러 객체 + 위치 정보까지. :contentReference[oaicite:11]{index=11} |
| 바운딩 박스(Bounding Box) | 객체를 감싸는 사각형 박스 | 보통 (x, y, w, h) 형태로 위치/크기를 나타냄 |
| mAP (mean Average Precision) | 평균 정밀도를 나타내는 지표 | 객체탐지 성능평가에서 많이 쓰임 |
| Precision (정밀도) | 모델이 예측한 객체 중 실제 객체인 비율 | TP / (TP + FP) |
| Recall (재현율) | 실제 객체 중 모델이 맞게 예측한 객체의 비율 | TP / (TP + FN) |
| FLOPs (Floating Point Operations) | 모델이 수행하는 연산량 | 연산량이 많을수록 일반적으로 느리거나 자원 많이 필요 |
| Anchor box | 사전에 정의된 여러 크기/비율의 박스 틀 | 객체탐지 시 예측을 쉽게 하기 위해 사용됨. YOLO 계열 모델마다 처리 방식이 달라질 수 있음 |
| Non-Max Suppression (NMS) | 여러 겹치는 박스 중에서 가장 적절한 하나만 남기는 후처리 기법 | 동일 객체에 여러 예측이 나왔을 때 정리하는 데 사용됨 |
| Epoch | 전체 학습 데이터셋을 한 번 모두 사용하는 학습 반복 횟수 | 학습할 때 보통 여러 epoch 동안 모델을 최적화함 |
| 학습률(Learning Rate) | 모델 가중치를 업데이트할 때 적용되는 스텝 크기 | 너무 크면 발산, 너무 작으면 너무 느림 |
| 커스텀 데이터셋(Custom Dataset) | 일반적인 공개 데이터셋이 아닌, 사용자가 직접 준비한 데이터셋 | 객체클래스, 이미지형태, 라벨링 방식 등이 다르므로 준비 잘 해야 함 |

## 4. 깃허브 레포 README 예시  
```markdown
# YOLOv11 실습 프로젝트  
**작성자**: [여러분 이름]  
**날짜**: YYYY-MM-DD  
**환경**: Python 3.x, PyTorch, Ultralytics YOLO  
**목표**: YOLOv8 / YOLOv11 비교 + 커스텀 데이터셋 적용 + 결과 시각화  
---

## 목차  
1. 프로젝트 소개  
2. 버전 비교 (YOLOv8 vs YOLOv11)  
3. 환경 설정  
4. 데이터셋 준비  
5. 모델 학습  
6. 결과 및 평가  
7. 느낀 점 & 향후 과제  
8. 참고자료  
9. 용어정리  

## 2. 버전 비교  
위 3장에서 정리한 표 삽입

## 3. 환경 설정  
```bash
# 가상환경 생성 (예시)
conda create -n yolov python=3.9
conda activate yolov

# 필요 라이브러리 설치
pip install ultralytics
````

## 4. 데이터셋 준비

* 데이터 폴더 구조 예시:

```
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

* `data.yaml` 예시:

```yaml
train: ./dataset/images/train
val: ./dataset/images/val
nc: 2
names: ['class1', 'class2']
```

## 5. 모델 학습

```bash
# YOLOv11 사용 예:
yolo detect train data=data.yaml model=yolo11l.pt epochs=50 imgsz=640
```

## 6. 결과 및 평가

* 학습 로그 및 결과 폴더: `runs/detect/train/`
* 평가 명령 예:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source='test.jpg'
```

* 결과 이미지 삽입 및 mAP / precision / recall 값 정리

## 7. 느낀 점 & 향후 과제

* 예: “YOLOv11은 YOLOv8에 비해 속도가 약간 빠르지만 내 데이터셋에서는 작은 객체에 대한 탐지율이 더 낮게 나왔다.”
* 예: “다음에는 세그멘테이션까지 적용해보겠다.”

## 8. 참고자료

* Ultralytics 공식 문서
* 비교 논문 및 블로그 글 등

