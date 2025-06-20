#!/bin/bash
set -euo pipefail

# 이 스크립트가 위치한 디렉토리 (프로젝트 루트)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# modules 패키지 인식용
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 처리할 백본 목록: vit, resnet (기본 둘 다)
BACKBONES=( vit resnet )

# 처리할 이미지 목록
IMAGES=(
  "$SCRIPT_DIR/Visual_Demo/images/cat.jpeg"
  "$SCRIPT_DIR/Visual_Demo/images/cat2.jpeg"
  "$SCRIPT_DIR/Visual_Demo/images/cat3.jpeg"
)

# 처리할 사이즈 목록
SIZES=(480 384 320 256)

# C++ 실행기 경로
EXE="$SCRIPT_DIR/CPP_Project/Feature_Extractor/build/trt_feature_extractor"

echo "▶ Starting feature comparison pipeline"
echo

for BACKBONE in "${BACKBONES[@]}"; do
  if [ "$BACKBONE" = "vit" ]; then
    CKPT_DIR="$SCRIPT_DIR/models/weights/ViT"
    PREFIX="lseg_img_enc_vit"
  else
    CKPT_DIR="$SCRIPT_DIR/models/weights/Resnet"
    PREFIX="lseg_img_enc_rn101"
  fi

  echo "▶ Processing backbone: $BACKBONE"
  echo "  • ckpt dir: $CKPT_DIR"
  echo

  # 1) Python으로 feature map 생성 (모든 ckpt, 모든 이미지, 모든 사이즈)
  echo "  → Generating PyTorch feature maps via model_output.py"
  python3 "$SCRIPT_DIR/python_trt_comp/model_output.py" \
    --weights_dir "$SCRIPT_DIR/models/weights" \
    --backbones "$BACKBONE" \
    --images "${IMAGES[@]}" \
    --sizes "${SIZES[@]}"
  echo

  # 2) 각 checkpoint 별로 TRT C++ feature extraction
  for CKPT in "$CKPT_DIR"/*.ckpt; do
    [ -e "$CKPT" ] || { echo "  [WARN] No ckpt files in $CKPT_DIR"; break; }
    NAME="$(basename "$CKPT" .ckpt)"
    echo "  • Checkpoint: $NAME.ckpt"

    # 엔진 파일 찾기 (weight 이름 그대로)
    PAT="${PREFIX}_${NAME}__*.trt"
    ENGINES=( $SCRIPT_DIR/models/trt_engines/$PAT )
    if [ ${#ENGINES[@]} -eq 0 ]; then
      echo "    [ERROR] No TRT engine found for pattern: $PAT"
      exit 1
    fi
    TRT_ENGINE="${ENGINES[0]}"
    echo "    – Using engine: $(basename "$TRT_ENGINE")"

    # C++ feature extractor 실행
    for IMG in "${IMAGES[@]}"; do
      IMG_STEM="$(basename "$IMG")"
      for SIZE in "${SIZES[@]}"; do
        echo "      > $EXE $TRT_ENGINE $IMG $SIZE"
        "$EXE" "$TRT_ENGINE" "$IMG" "$SIZE"
      done
    done
    echo
  done

done

echo "✅ All comparisons done!"
