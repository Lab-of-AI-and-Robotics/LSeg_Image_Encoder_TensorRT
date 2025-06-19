#!/bin/bash
set -euo pipefail

# 이 스크립트 파일이 있는 디렉토리 → 곧 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PYTHONPATH 에 프로젝트 루트 추가 (modules 패키지 인식용)
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 이미지 파일 리스트 (Visual_Demo 폴더 아래)
IMAGES=(
  "$SCRIPT_DIR/Visual_Demo/images/cat.jpeg"
  "$SCRIPT_DIR/Visual_Demo/images/cat2.jpeg"
  "$SCRIPT_DIR/Visual_Demo/images/cat3.jpeg"
)

# 체크포인트 리스트
WEIGHTS=(
  "$SCRIPT_DIR/models/weights/demo_e200.ckpt"
  #"$SCRIPT_DIR/models/weights/fss_l16.ckpt"
)

# Weight → Tag 변환 함수
get_tag_from_weight() {
  local w=$(basename "$1")
  if [[ "$w" == *demo* ]]; then
    echo ade20k
  elif [[ "$w" == *fss* ]]; then
    echo fss
  else
    echo custom
  fi
}

# 사용할 이미지 크기
SIZES=(480 384 320 260)

for IMG in "${IMAGES[@]}"; do
  for WEIGHT in "${WEIGHTS[@]}"; do
    TAG=$(get_tag_from_weight "$WEIGHT")

    echo "✅ Running PyTorch Model (Weight: $TAG) for image: $IMG"
    python3 "$SCRIPT_DIR/python_trt_comp/model_output.py" \
      --weights "$WEIGHT" \
      --image "$IMG" \
      --sizes "${SIZES[@]}"

    echo "✅ Running TensorRT Extraction (Weight: $TAG) for image: $IMG"
    EXE="$SCRIPT_DIR/CPP_Project/Feature_Extractor/build/trt_feature_extractor"

    # 패턴에 맞는 첫 번째 trt 엔진을 찾아서 (글로브 확장 허용)
    ENGINES=( $SCRIPT_DIR/models/trt_engines/lseg_img_enc_vit_${TAG}__*.trt )
    if [ ${#ENGINES[@]} -eq 0 ]; then
      echo "[ERROR] No TRT engine found for tag '$TAG'!"
      exit 1
    fi
    TRT_ENGINE=${ENGINES[0]}

    for SIZE in "${SIZES[@]}"; do
      echo "[INFO] $EXE $TRT_ENGINE $IMG $SIZE"
      "$EXE" "$TRT_ENGINE" "$IMG" "$SIZE"
    done
  done

  echo "✅ Comparing Features for image: $IMG"
  python3 "$SCRIPT_DIR/python_trt_comp/compare_features.py" \
    --image "$IMG" \
    --sizes "${SIZES[@]}"
done

echo "✅ All comparions done!"
