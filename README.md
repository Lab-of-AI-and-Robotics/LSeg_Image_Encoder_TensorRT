# LSeg Image Encoder – ONNX & TensorRT

# *아직 작업중입니다! 완료되지 않았습니다!!!*

> ✨ **End‑to‑end pipeline for converting the **LSeg** image encoder to ONNX / TensorRT, benchmarking      PyTorch ↔ TRT speed, and verifying numerical fidelity.**

---

## 0. TL;DR  (run everything)

```bash
# (1) install python deps
pip install -r requirements.txt  # CUDA / OpenCV must already be available

# (2) build both C++ projects in one shot
make             # or   make -j12

# (3) optional – run the latency benchmark
python3 inferenceTimeTester.py  \
  --weights models/weights/demo_e200.ckpt \
  --img_sizes 260 390 520 650 780 910

# (4) run the full feature‑comparison pipeline
bash python_trt_comp/run_feature_comparison.sh


```

* `make clean`  → removes **all** `CPP_Project/**/build` directories + temporary CMake artefacts.
* The **root‑level `Makefile`** is just a thin wrapper that `cmake --build`’s each sub‑directory – it does **not** introduce any extra dependencies.

---

## 1. Installation
0. **System Package**
```bash
sudo apt update && sudo apt install -y \
    python3-pip python3-dev \
    libopencv-dev \
    libprotobuf-dev protobuf-compiler \
    libtinfo5 \
    libopenmpi-dev \
    cuda-toolkit-##  # CUDA Version : At least Minimum Requirement for TensorRT 10.9
```
1.  **Python** ≥3.8   `pip install -r requirements.txt`
2.  **CUDA + TensorRT 10.9** already installed (the repo never calls the TRT builder directly – it simply links against the headers/lib).
3.  Optional but recommended:  `opencv-dev` for the C++ extractor.

---

## 2. Building  

### 2‑a. One‑shot build (recommended)

```bash
# from project root
make -j$(nproc)        # builds …
                      #   • CPP_Project/Inference_Time_Tester
                      #   • CPP_Project/Feature_Extractor
```
The helper `Makefile` simply iterates through every `CPP_Project/*/CMakeLists.txt`, wipes the old `build/` directory, configures with `cmake -S . -B build`, then invokes the native generator.

### 2‑b. Per‑project build (legacy)

```bash
cd CPP_Project/Inference_Time_Tester && cmake -B build -S . && cmake --build build -j
cd CPP_Project/Feature_Extractor     && cmake -B build -S . && cmake --build build -j
```

---

## 3. Project Structure

```
LSeg_Image_Encoder_TensorRT/
│
├── CPP_Project/                      # C++ 프로그램 모음
│   ├── Feature_Extractor/            # Feature 추출기 프로젝트
│   │   ├── CMakeLists.txt            # CMake 설정
│   │   └── main.cpp                  # Feature 추출기 메인 코드
│   ├── Inference_Time_Tester/        # Inference 벤치마크 프로젝트
│   │   ├── CMakeLists.txt            # CMake 설정
│   │   └── main.cpp                  # 벤치마크 메인 코드
│   └── third_party/                  # C++ 서드 파티
│       └── cnpy/                     # CNpy submodule for numpy I/O
│
├── Visual_Demo/                      # 데모 스크립트 및 결과
│   ├── demo.sh                       # demo.sh 스크립트
│   ├── demo.py                       # demo.py 호출 래퍼
│   ├── demo_wordFree.sh              # demo_wordFree.sh 스크립트
│   ├── demo_wordFree.py              # demo_wordFree.py 호출 래퍼
│   └── images/                       # 시각화 결과 및 입력 이미지
│       ├── Dog_grass_demo.png        # segmentation 결과 예시
│       ├── Dog_grass_wordFree.png    # word-free 결과 예시
│       └── dog_grass.jpeg            # 입력 이미지 예시
│
├── models/
│   ├── weights/
│   │   ├── demo_e200.ckpt # 예시: ViT-L/16 CLIP 모델 체크포인트
│   │   └── fss_l16.ckpt   # 예시: FSS 기반 ViT-L/16 모델 체크포인트
│   ├── onnx_engines/
│   │   ├── lseg_img_enc_vit_ade20k.onnx  # 예시 ONNX 모델
│   └── trt_engines/
│        │     # 예시 TRT 엔진
│        └── lseg_img_enc_vit_ade20k__fp16_sparse_ws512MiB.trt 
│
├── modules/               # LSeg 모델 관련 소스
│   ├── lseg_module.py     # LSegModule: 이미지 인코더 + 헤드 래핑
│   ├── lseg_full.py       # LSegFull: 백본과 헤드 포함 전체 네트워크
│   ├── models/            # 내부 서브모듈
│        ├── lseg_blocks.py  # RefineNet 블록과 skip-connection 처리
│        ├── lseg_net.py     # 네트워크 assemble 유틸리티
│        └── lseg_vit.py     # CLIP ViT 레이어 분할 및 feature 추출
│
├── conversion/
│   ├── model_to_onnx.py   # PyTorch → ONNX 변환 스크립트
│   ├── onnx_to_trt.py     # ONNX → TensorRT 변환 스크립트
│   └── onnx_to_trt.sh     # 쉘 래퍼 스크립트
│
├── CPP_Project/
│   └── Inference_Time_Tester/      # C++ 벤치마크 프로젝트
│       ├── CMakeLists.txt          # CMake 설정
│       ├── main.cpp                # 벤치마크 메인 코드
│       └── build/                  # 빌드 결과물 (trt_cpp_infer_time_tester 실행파일)
│
├── python_trt_comp/                  # Python 기반 Feature 비교 스크립트
│   ├── compare_features.py           # Feature map 비교 (Cosine/L2)
│   ├── compare_inputs.py             # 입력 tensor 비교
│   ├── model_output.py               # PyTorch Feature 추출 스크립트
│   └── run_feature_comparison.sh     # 전체 파이프라인 실행 스크립트
│
├── inferenceTimeTester.py # 추론 및 벤치마크 메인 스크립트 (루트 폴더)
│
├── requirements.txt       # Python 패키지 목록
├── Makefile                    # new – one‑shot builder wrapper ❶
└── README.md              # 본 파일
```

---
## 4. 모델 다운로드

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
## 5. Build ONNX & TensorRT Engine

### 1) ONNX 모델 변환

```bash
python3 conversion/model_to_onnx.py \
  --weights models/weights/demo_e200.ckpt \
```

* `--weights`: 체크포인트 경로

### 2) TensorRT 엔진 변환
**주의**: TensorRT 변환은 GPU 및 환경에 따라 다르므로, **실행할 기기에서 직접 변환**해야 합니다.
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

## 6.  Latency Benchmark

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

### 6‑a. Test rig

```
AMD Ryzen 7 9700X  | 8C / 16T @ 5.0 GHz
NVIDIA RTX 4090    | 24 GB (Ada, 550 W limit)
64 GB DDR5‑6000    | dual‑rank
TensorRT 10.9 + CUDA 12.2, PyTorch 2.3 (cu118)
Ubuntu 22.04 LTS   | Linux 6.5
```

Hardware script (`hardware_spec.sh`) dumps the table automatically.

### 6‑b. Results  
`inferenceTimeTester.py --iterations 1000`

| Size | PyTorch **ms** |  ± | TRT‑Python **ms** | ± | TRT‑C++ **ms** | ± |
|-----:|---------------:|----:|------------------:|---:|---------------:|---:|
| 260 | 10.72 | 0.78 | 5.66 | 0.16 | **4.10** | 0.20 |
| 390 | 20.03 | 1.49 | 8.60 | 0.31 | **4.97** | 0.34 |
| 520 | 35.42 | 2.88 | 14.76 | 0.27 | **8.26** | 0.34 |
| 650 | 56.91 | 4.67 | 21.82 | 0.51 | **11.21** | 1.30 |
| 780 | 86.86 | 6.22 | 32.18 | 0.56 | **17.67** | 0.91 |
| 910 | 136.54 | 11.55 | 45.71 | 1.95 | **24.85** | 1.08 |

**Observations**

* TensorRT ( Python API ) already yields a **2 – 3× speed‑up** over eager PyTorch.
* The minimalist C++ runner shaves **another ~40 % latency**, dominated by
  * avoiding `pycuda` / DLPack marshalling overheads;
  * pre‑parsing I/O tensor indices at start‑up.
* Slope ≈ O(N²) w.r.t spatial resolution (expected for ViT windowed attention).

---

### 7. Demo Scripts

### Visual_Demo/demo.sh

이 스크립트는 **ONNX 모델**을 이용해 예시 이미지를 분할(segmentation)하고, 결과를 시각화합니다.

```bash
# 사용 예시 (루트에서 실행)
python3 Visual_Demo/demo.py --image Visual_Demo/images/dog_grass.jpeg \
                            --labels "dog, grass, other" \
                            --onnx models/onnx_engines/lseg_img_enc_vit_ade20k.onnx \
                            --size 384
```

* `--image`: 입력 이미지 경로
* `--labels`: 콤마(,)로 구분된 라벨 목록 (예: "cat, sky, building")
* `--onnx`: ONNX 모델 파일 경로
* `--size`: 모델 입력 크기 (HxW)

스크립트 내부에서는 `demo.py` 를 호출하며, 좌측에 원본 이미지, 우측에 segmentation 결과를 출력합니다.


### Visual_Demo/demo\_wordFree.sh

이 스크립트는 **Full CLIP Vocab 기반** 픽셀 단위 분류를 수행하고, **등장한 단어**를 콘솔에 출력하며 시각화합니다.

```bash
# 사용 예시 (루트에서 실행)
python3 Visual_Demo/demo_wordFree.py --image Visual_Demo/images/dog_grass.jpeg \
                                     --onnx models/onnx_engines/lseg_img_enc_vit_ade20k.onnx \
                                     --size 384
```

* `--image`: 입력 이미지 경로
* `--onnx`: ONNX 모델 파일 경로
* `--size`: 모델 입력 크기 (HxW)

스크립트 내부에서는 `demo_wordFree.py` 를 호출하여 이미지 내 픽셀마다 CLIP 전체 vocab 중 가장 유사도가 높은 토큰을 선택, 해당 단어들을 출력하고 결과를 시각화합니다.

### Visual Results

아래는 `Visual_demo/images/` 폴더에 저장된 예시 결과입니다:

|                     Segmentation (`demo.py`)                     |                  Word-free (`demo_wordFree.py`)                  |      |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | ---- |
| ![Dog Grass Segmentation](Visual_Demo/images/Dog_grass_demo.png) | ![Dog Grass WordFree](Visual_Demo/images/Dog_grass_wordFree.png) |

---

## 8.  Feature‑map Extraction & Comparison

| **Script / Binary** | **Role** | **Backend** |
|---------------------|-------------------------------------------|-----------|
| `python_trt_comp/model_output.py` | Loads an LSeg checkpoint, **removes the decoder**, runs the encoder only and dumps a   `(B,512,H/2,W/2)` feature‑tensor to `npy`. | PyTorch |
| `CPP_Project/Feature_Extractor/build/trt_feature_extractor` | Deserialises the dynamic‑shape **TensorRT engine**, feeds a BGR image, and writes the identical feature‑tensor. | TensorRT C++ |
| `python_trt_comp/compare_features.py` | Loads both tensors, flattens them, outputs **cosine similarity** & **L2 norm**. | PyTorch (CPU) |
| `python_trt_comp/run_feature_comparison.sh` | Glue: loops over several images × checkpoints × resolutions. | bash |

전체 Feature 추출 및 비교 파이프라인 실행:

```bash
bash python_trt_comp/run_feature_comparison.sh
```

* `run_feature_comparison.sh`: `model_output.py`, C++ Feature Extractor, `compare_features.py` 순차 실행
* 결과는 `outputs/` 폴더와 콘솔 로그로 확인합니다.

---

### 8‑a. Numerical‑fidelity results  
*(RTX 4090 + TensorRT 10.9 – FP16 engine, sparse weights)*

| **Size** | **Cosine (↑) – PyTorch vs TRT** | **L2 (↓)** |
|-------:|---------------:|-----------:|
| 480 | **1.0007** | 1.43 |
| 384 | **1.0010** | 5.67 |
| 320 | **1.0012** | 3.91 |
| 260 | **1.0007** | 1.26 |

*Cosine ≈ 1.0* implies that FP16 quantisation + builder optimisations introduce **<0.1 % angular error** – well within acceptable limits for CLIP‑style similarity retrieval.

### 8-b 📊 Cross‑Tag Feature Map Comparison (ade20k vs fss)

| Size (px) | Framework | Cosine Similarity | L2 Distance |
|-----------|-----------|------------------:|------------:|
| **480** | PyTorch | **‑0.025655** | **343.642** |
|           | TensorRT C++ | ‑0.026256 | 343.743 |
| **384** | PyTorch | **‑0.013745** | **273.327** |
|           | TensorRT C++ | ‑0.014547 | 273.434 |
| **320** | PyTorch | **‑0.004119** | **226.718** |
|           | TensorRT C++ | ‑0.004628 | 226.775 |
| **260** | PyTorch | **‑0.003275** | **181.305** |
|           | TensorRT C++ | ‑0.003252 | 181.303 |

> *Negative cosine similarity indicates that the aggregated visual embeddings for **ade20k** and **fss** tags are nearly orthogonal, reflecting the distinct semantic domains of the two training sets.  
> The L2 distances corroborate this, staying consistently in the 180‒340 range across spatial scales.  
> TensorRT outputs track PyTorch extremely closely (< 0.001 absolute delta in cosine; < 0.1 % in L2), confirming numerical parity after quantisation and kernel fusion.*

### 8-c Visual Results

아래는 `Visual_demo/images/` 폴더에 저장된 예시 결과입니다:
![PT TRT Comparison](Visual_Demo/images/pt_trt_comp_cat1.png)

---

## 9. Additional Notes

* **ONNX opset\_version=14** 사용
* 동적 입력 크기 지원: `torch.onnx.export(... dynamic_axes=...)` 설정 참조
* GPU 벤치마크를 위해 `onnxruntime-gpu` 필요: `pip install onnxruntime-gpu`
* CUDAExecutionProvider 확인:

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

---

## 10. License

MIT – see `LICENSE` for details.

---

## 11. Acknowledgements

Portions of the code are adapted from **ISL‑org / lang‑seg** (Apache‑2.0) and **NVIDIA TensorRT samples**.
