# OCR 개발 프로젝트 스캐폴드

이 워크스페이스는 전달받은 OCR 흐름도를 기준으로 구성한 프로젝트 뼈대입니다.  
한국어 근무표 OCR 파이프라인을 단계별로 개발할 수 있도록 데이터 수집, 전처리, 모델 설계, 학습, 평가, 배포 영역을 나누어 정리했습니다.

## 단계 구성

1. `stage1_data`: 데이터 수집 기준, 합성 데이터 생성, 어노테이션 가이드, 데이터셋 분할/검수
2. `stage2_preprocess`: 이미지 전처리, 데이터 증강, 한국어 문자 사전
3. `stage3_models`: Detection, Recognition, Angle Classifier, 전이학습 전략
4. `stage4_training`: 학습 환경, loss/optimizer, 하이퍼파라미터, 모니터링
5. `stage5_evaluation`: Detection 평가, Recognition 평가, E2E 평가, 품질 게이트
6. `stage6_deployment`: 모델 export, 서비스 통합, 신뢰도 UI, 버전 관리

## 실행 시작점

현재 스캐폴드 요약은 아래 명령으로 확인할 수 있습니다.

```bash
python -m ocr_project.main
```

## 빠른 설치

### 공통 실행용 최소 라이브러리

프로젝트 초기 구조 확인과 로컬 유틸리티 실행에 필요한 최소 패키지입니다.

```bash
pip install numpy opencv-python pyyaml pillow
```

### OCR 학습용 전체 환경

Detection, Recognition, Angle Classifier 학습까지 포함한 전체 학습 스택은 아래처럼 설치합니다.

```bash
pip install -r requirements_train.txt
```

`requirements_train.txt`를 쓰지 않고 한 번에 직접 설치하려면:

```bash
pip install paddleocr==2.8.0 opencv-python==4.9.0.80 albumentations==1.4.0 wandb==0.16.0 pyyaml==6.0.1 tqdm==4.66.0 matplotlib==3.8.0 scikit-learn==1.4.0 shapely==2.0.3 lmdb==1.4.1
```

### PaddlePaddle 설치

학습 전에 환경에 맞는 PaddlePaddle을 별도로 설치해야 합니다.

GPU 환경 예시:

```bash
python -m pip install paddlepaddle-gpu==2.6.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu117/
```

CPU 환경 예시:

```bash
python -m pip install paddlepaddle==2.6.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

### 한 번에 세팅하는 예시

새 학습 환경을 가장 빠르게 만드는 기본 순서는 아래와 같습니다.

```bash
python -m venv .venv_train
.venv_train\Scripts\activate
python -m pip install --upgrade pip
python -m pip install paddlepaddle-gpu==2.6.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu117/
pip install -r requirements_train.txt
```

## 핵심 개념

이 프로젝트는 단일 OCR 모델 하나가 아니라, 근무표 이미지를 안정적으로 읽기 위한 전체 OCR 파이프라인입니다.

### 1. Detection과 Recognition의 차이

- Detection은 이미지 안에서 텍스트가 어디에 있는지 찾습니다.
- Recognition은 Detection이 잘라낸 텍스트 영역을 실제 문자열로 읽습니다.
- Detection이 놓친 텍스트는 Recognition이 아무리 좋아도 읽을 수 없습니다.

이 프로젝트에서의 기본 선택은 다음과 같습니다.

- Detection 모델: `DBNet++`
- Recognition 모델: `SVTR-Tiny`
- Recognition 대안 모델: `CRNN`
- Angle Classifier: `MobileNetV3-Small`

### 2. 전처리가 중요한 이유

실제 근무표 이미지는 항상 깔끔하지 않습니다.

- 스마트폰 촬영본은 기울어져 있을 수 있습니다.
- 스캔본은 대비가 낮을 수 있습니다.
- 화면 캡처는 압축 열화가 있을 수 있습니다.
- 손글씨와 인쇄 텍스트가 섞일 수 있습니다.

그래서 학습과 추론 전에 입력을 가능한 한 표준 형태로 맞춰야 합니다.  
이 프로젝트의 전처리 파이프라인은 아래 단계를 중심으로 설계되어 있습니다.

- 이미지 로드 및 포맷 정규화
- 방향 보정
- 노이즈 제거 및 선명화
- 기울기 보정
- 리사이즈 및 픽셀 정규화

### 3. 증강이 중요한 이유

실제 데이터는 항상 부족하고, 특정 포맷이나 업종에 치우치기 쉽습니다.  
증강은 데이터를 더 다양하게 만들어 모델이 실제 서비스 입력에 강해지도록 돕습니다.

- Detection 증강은 전체 이미지와 바운딩박스를 함께 바꿉니다.
- Recognition 증강은 잘린 crop 이미지에 더 약한 변형을 적용합니다.
- Angle Classifier 증강은 라벨 의미가 바뀌지 않도록 회전 계열 변형을 조심해서 다룹니다.

### 4. 문자 사전의 역할

Recognition 모델은 아무 글자나 출력하는 것이 아닙니다.  
문자 사전에 있는 글자만 예측할 수 있습니다.

- 현대 한글 완성형
- 영문 대문자/소문자
- 숫자
- 근무표 특화 기호
- `<unk>`, `<pad>`, `<blank>` 같은 특수 토큰

만약 실제 근무표에 등장하는 기호가 사전에 없으면, 모델은 그 글자를 정확히 출력할 수 없습니다.

### 5. 왜 단계별 파이프라인으로 나누는가

이 프로젝트는 흐름도 기반의 단계형 구조를 따릅니다. 이유는 각 단계 결과가 다음 단계 품질을 직접 결정하기 때문입니다.

1. 데이터 수집 및 마스킹
2. 합성 데이터 생성
3. 어노테이션
4. 데이터셋 분할 및 품질 검수
5. 전처리 및 증강
6. 모델 설계
7. 학습 환경 및 학습 설정
8. 평가
9. 배포

이렇게 나누면 문제 원인을 추적하기 쉬워집니다.  
예를 들어 Recognition 성능이 낮아 보여도 실제 원인은 잘못된 crop, 누락된 문자 사전, 부정확한 라벨일 수 있습니다.

## 주요 기술 스택

### PaddlePaddle / PaddleOCR

- `PaddlePaddle`은 이 프로젝트의 딥러닝 학습/추론 프레임워크입니다.
- `PaddleOCR`은 DBNet++, SVTR, MobileNetV3 같은 OCR 관련 기준 구현을 제공합니다.
- 이 프로젝트는 PaddleOCR 생태계 위에서 커스텀 구조를 쌓는 방향으로 설계되어 있습니다.

### OpenCV

OpenCV는 다음 작업에 사용됩니다.

- 이미지 로드
- 리사이즈
- 노이즈 제거
- 블러 및 샤프닝
- 기하학적 변환
- 시각화와 박스 드로잉

### NumPy

NumPy는 다음 작업에 사용됩니다.

- 이미지 배열 처리
- 맵 생성
- 정규화
- 배치 패딩 및 텐서 변환

### YAML 설정 파일

주요 동작은 코드 하드코딩보다 YAML 설정으로 제어합니다.

- 모델 설정
- 전처리 설정
- 증강 설정
- 학습 설정

이 방식은 실험 재현성과 설정 비교에 유리합니다.

### LMDB

LMDB는 대용량 OCR 데이터에서 선택적으로 사용하는 I/O 최적화 수단입니다.

- crop 파일이 많아질수록 파일 시스템 병목을 줄일 수 있습니다.
- Recognition 학습에서 특히 효과적일 수 있습니다.
- 처음부터 필수는 아니고, DataLoader가 병목일 때 도입하면 됩니다.

### pretrained weight와 checkpoint의 차이

둘은 비슷해 보이지만 역할이 다릅니다.

- pretrained weight: 사전학습된 초기 가중치
- checkpoint: 내 학습 과정 중 저장한 진행 상태

pretrained weight는 좋은 초기값을 주고, checkpoint는 학습 중단 후 이어서 학습하거나 최고 성능 모델을 보존하는 데 필요합니다.

## 미리 알면 좋은 배경지식

이 저장소를 편하게 다루려면 아래 개념을 알고 있으면 좋습니다.

- Python 패키징과 가상환경
- OpenCV 기반 기본 컴퓨터 비전 처리
- supervised learning 데이터셋 구성과 train/val/test 분할
- OCR에서의 Detection / Recognition 구분
- CTC 디코딩 기초
- GPU 메모리, 배치 크기, mixed precision
- YAML 기반 실험 설정 관리

전부 미리 알고 시작할 필요는 없지만, 이 개념들이 있으면 학습과 디버깅 흐름을 훨씬 빨리 이해할 수 있습니다.

## 현재 구현 범위

### OCR-D01 데이터 수집 기준

`stage1_data/collection_spec.py`에는 근무표 원본 수집 목표, 품질 기준, 개인정보 처리 규칙, 폴더 구조 기준이 들어 있습니다.

`stage1_data/collection_workflow.py`는 아래 기능을 제공합니다.

- `data/raw`, `data/masked`, `data/rejected`, `data/meta` 초기화
- 명명 규칙에 맞는 파일명 생성 및 검증
- 이미지 품질 합격/탈락 판정
- `collection_log.csv`, `quality_check.csv` 기록 보조

### OCR-D02 합성 데이터 생성

`stage1_data/synthetic_generator.py`는 아래 기능을 제공합니다.

- `data/synthetic/render`, `data/synthetic/aug`, `data/synthetic/labels` 구조 초기화
- 렌더링/증강 결과용 파일명 생성
- 기본 폰트/사전/노이즈 설정
- 랜덤 시드 기반 `synth_params.json` 재현 가능 설정
- 생성 수량 점검 및 로그 기록 보조

### OCR-D03 어노테이션 가이드

`stage1_data/annotation_guide.py`와 `stage1_data/annotation_workflow.py`는 아래 기능을 제공합니다.

- PPOCRLabel 중심 도구 추천
- bbox 및 텍스트 입력 규칙
- `data/labels` 작업공간 초기화
- PaddleOCR `det_gt.txt`, `rec_gt.txt` 포맷 생성 보조
- IoU 기반 캘리브레이션 점검

### OCR-D04 데이터셋 분할 및 검수

`stage1_data/dataset_splitter.py`는 아래 기능을 제공합니다.

- masked / synthetic / labels 결과물 선행조건 확인
- 자동 품질 점검
- 계층적 train/val/test 분할
- 데이터 누수 및 difficult 제외 규칙 검증
- `split_config.json`, `dataset_stats.json`, `quality_report.md` 생성 보조

### OCR-P01 전처리 모듈

`stage2_preprocess/preprocess.py`는 아래 기능을 제공합니다.

- 학습/추론 공용 `PreprocessPipeline`
- 로드, 방향 보정, 노이즈 제거, deskew, resize, normalize 단계 제어
- Detection/Recognition 별 입력 규격 처리
- 시각화 결과 저장

### OCR-P02 온라인 증강

`stage2_preprocess/augmentation.py`는 아래 기능을 제공합니다.

- 학습 전용 `AugmentPipeline`
- Detection용 기하/광학 변환과 bbox 동기화
- Recognition용 `single_char`, `date`, `handwrite`, `normal` 증강 분기
- 배치 처리와 시각화 보조

### OCR-P03 한국어 문자 사전

`stage2_preprocess/korean_charset.py`는 아래 기능을 제공합니다.

- `rec_gt.txt` 기반 문자 추출
- PaddleOCR 호환 문자 사전 생성
- 메타 정보 및 빈도 통계 파일 작성
- 커버리지, 중복, UTF-8, `<blank>` 위치 검증

### OCR-M01 Detection 모델

`stage3_models/detection_model.py`와 `backend/ocr/models/det/`는 아래 기능을 제공합니다.

- ResNet50 + FPNASF + DBHead 기반 DBNet++ 스펙
- `(1, 3, 960, 960) -> (1, 1, 240, 240)` 출력 shape 시뮬레이션
- freeze/unfreeze 전략 요약
- 후처리와 padding 역변환 보조

### OCR-M02 Recognition 모델

`stage3_models/recognition_model.py`와 `backend/ocr/models/rec/`는 아래 기능을 제공합니다.

- SVTR-Tiny 기본 스펙
- CRNN 대안 스펙
- CTC 디코더 구성
- `(4, 3, 32, 128) -> (4, 32, vocab_size)` shape 시뮬레이션

### OCR-M03 Angle Classifier

`stage3_models/angle_classifier.py`와 `backend/ocr/models/cls/`는 아래 기능을 제공합니다.

- MobileNetV3-Small 기반 0도/180도 분류기 스펙
- `data/angle/{train,val,test}/{0,1}` 데이터셋 구조 생성 보조
- 임계값 기반 회전 판단 규약
- `(1, 3, 48, 192) -> (1, 2)` shape 점검

### OCR-T01 학습 환경

`stage4_training/environment.py`, `requirements_train.txt`, `train/` 디렉토리는 아래 기능을 제공합니다.

- RTX 2080 Ti 11GB 기준 학습 기본값
- Detection / Recognition / Classifier용 학습 설정 파일
- DataLoader 초안, trainer 초안, 스크립트 초안
- checkpoint, log, weights, LMDB 변환 구조

## 학습 환경 기본값 메모

현재 사용자가 제공한 GPU 정보 기준으로 기본 학습값은 아래 방향으로 맞춰져 있습니다.

- GPU: `RTX 2080 Ti 11GB`
- 기본 precision: `fp16`
- 기본 사용 GPU: `gpu_ids: [0]`
- Classifier batch size: `128`
- Detection batch size: `12`
- Recognition batch size: `64`

이 값들은 안정적으로 첫 학습을 시작하기 위한 보수적 기준입니다.  
실제 학습에서 여유 VRAM이 확인되면 Detection 배치 크기부터 점진적으로 늘리는 방식이 안전합니다.
