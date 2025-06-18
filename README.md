# LSeg Image Encoder ONNX & TensorRT

# *아직 작업중입니다! 완료되지 않았습니다!!!*

본 프로젝트는 **LSeg 모델의 이미지 인코딩 경로**(백본, 중간 feature 추출, projection layer)를 분리하여 **ONNX** 및 **TensorRT** 모델로 변환하고, 각 단계별 **Inference 성능 비교**를 지원합니다.

---

## Installation

### 1. Python 환경

* Python 3.8 이상 권장
* 가상환경(venv, conda 등) 사용을 권장합니다.

### 2. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 3. 시스템 패키지 설치 (Ubuntu 예시)

```bash
sudo apt update && sudo apt install -y \
    python3-pip python3-dev \
    libopencv-dev \
    libprotobuf-dev protobuf-compiler \
    libtinfo5 \
    libopenmpi-dev \
    cuda-toolkit-##  # CUDA 설치 필요시 버전에 맞춰
```

---

## 모델 다운로드

**Weight 파일**은 LSeg 공식 저장소에서 가져오세요.
LSeg 공식 저장소 : **[https://github.com/isl-org/lang-seg](https://github.com/isl-org/lang-seg)**

```bash
# 메인 ViT-L/16 모델 (demo_e200.ckpt)
pip install gdown
# demo_e200.ckpt 다운로드
gdown 'https://drive.google.com/uc?id=1FTuHY1xPUkM-5gaDtMfgCl3D0gR89WV7'

# FSS 데이터셋 기반 모델 예시
# fss_rn101.ckpt (ResNet101)
gdown 'https://drive.google.com/uc?id=1UIj49Wp1mAopPub5M6O4WW-Z79VB1bhw'
# fss_l16.ckpt (ViT-L/16)
gdown 'https://drive.google.com/uc?id=1Nplkc_JsHIS55d--K2vonOOC3HrppzYy'
```

다운로드한 체크포인트는 `models/weights/` 아래에 저장하세요.

---

## Project Structure

```
LSeg_Image_Encoder_TensorRT/
├── models/
│   ├── weights/
│   │   ├── demo_e200.ckpt # 예시: ViT-L/16 CLIP 모델 체크포인트
│   │   └── fss_l16.ckpt   # 예시: FSS 기반 ViT-L/16 모델 체크포인트
│   ├── onnx_engines/
│   │   ├── lseg_img_enc_vit_ade20k.onnx  # 예시 ONNX 모델
│   └── trt_engines/
│        │     # 예시 TRT 엔진
│        ├── lseg_img_enc_vit_ade20k__fp16_sparse_ws512MiB.trt 
│
├── conversion/
│   ├── model_to_onnx.py   # PyTorch → ONNX 변환 스크립트
│   ├── onnx_to_trt.py     # ONNX → TensorRT 변환 스크립트
│   └── onnx_to_trt.sh     # 쉘 래퍼 스크립트
│
├── inferenceTimeTester.py # 추론 및 벤치마크 메인 스크립트 (루트 폴더)
│
├── modules/               # LSeg 모델 관련 소스
│   ├── lseg_module.py     # LSegModule: 이미지 인코더 + 헤드 래핑
│   ├── lseg_full.py       # LSegFull: 백본과 헤드 포함 전체 네트워크
│   ├── models/            # 내부 서브모듈
│   │   ├── lseg_blocks.py  # RefineNet 블록과 skip-connection 처리
│   │   ├── lseg_net.py     # 네트워크 assemble 유틸리티
│   │   └── lseg_vit.py     # CLIP ViT 레이어 분할 및 feature 추출
├── build/                 # C++ 빌드 결과 (trt_cpp_infer_time_tester)
│   └── trt_cpp_infer_time_tester # C++ TensorRT 벤치마크 실행 파일
├── requirements.txt       # Python 패키지 목록
├── CMakeLists.txt         # C++ 프로젝트 설정
└── README.md              # 본 파일
```

## Usage

### 1. ONNX 모델 변환

```bash
python3 conversion/model_to_onnx.py \
  --weights models/weights/demo_e200.ckpt \
```

* `--weights`: 체크포인트 경로

### 2. TensorRT 엔진 변환

```bash
python3 conversion/onnx_to_trt.py \
  --onnx models/onnx_engines/lseg_img_enc_vit_ade20k.onnx \
  --workspace 1073741824 \
  --fp16 \
  --sparse \
  --disable-timing-cache \
  --gpu-fallback \
  --debug 
```

#### TensorRT 옵션 설명

| 옵션                         | 종류      | 기본값     | 설명                           |
| -------------------------- | ------- | ------- | ---------------------------- |
| `--onnx <PATH>`            | 필수    | —      | 입력 ONNX 파일 경로                |
| `--workspace <BYTE>`       | integer | `1<<29` | 빌더 워크스페이스 메모리(바이트)           |
| `--fp16` / `--no-fp16`     | flag    |  true   | FP16 연산 사용 여부                |
| `--sparse` / `--no-sparse` | flag    |  true   | Sparse weights 전술 사용 여부      |
| `--disable-timing-cache`   | flag    | false   | 타이밍 캐시 비활성화 (빌드 안정성 ↑, 속도 ↓) |
| `--gpu-fallback`           | flag    | false   | INT8 모드에서 GPU 연산 폴백 허용       |
| `--debug`                  | flag    | false   | 디버그 로그 활성화                   |

**엔진 파일명 자동 생성 규칙**: `base__<옵션1>_<옵션2>_..._<wsXXMiB>.trt`

---

### 3. Inference & Benchmark

`inference/inferenceTimeTester.py` 를 실행하여 **PyTorch, ONNX, TensorRT** 속도를 비교합니다.

```bash
python3 inference/inferenceTimeTester.py \
  --weights models/weights/demo_e200.ckpt \
  --iterations 500 \
  --img_sizes 260 390 520 650 780 910 \
  --trt_fp16 \
  --trt_sparse \
  --trt_no_tc \
  --trt_gpu_fb \
  --trt_debug \
  --trt_workspace 1073741824
```

* `--img_sizes`: 테스트할 입력 크기 목록
* `--iterations`: 반복 횟수
* `--trt_*`: TRT 빌드 옵션 (ONNX→TRT에 자동 반영)

**스크립트 동작**:

1. ONNX 파일이 없으면 자동 생성
2. TRT 엔진이 없으면 자동 생성
3. PyTorch → ONNX → TRT 순으로 추론 벤치마크

**결과 예시**:

```
[RESULT] PyTorch Avg: 12.345 ms ± 0.123 ms
[RESULT] ONNX   Avg: 10.567 ms ± 0.098 ms
[RESULT] TRT    Avg:  5.432 ms ± 0.045 ms
```

---

## Additional Notes

* **ONNX opset\_version=14** 사용
* 동적 입력 크기 지원: `torch.onnx.export(... dynamic_axes=...)` 설정 참조
* GPU 벤치마크를 위해 `onnxruntime-gpu` 필요: `pip install onnxruntime-gpu`
* CUDAExecutionProvider 확인:

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

---

## License

사용 중인 라이선스를 명시하세요. 예: MIT License
