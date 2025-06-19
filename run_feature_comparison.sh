#!/bin/bash

# 이미지 파일 리스트
IMAGES=("images/cat.jpeg" "images/cat2.jpeg" "images/cat3.jpeg")
WEIGHTS=("modules/demo_e200.ckpt" "modules/fss_l16.ckpt")

# Weight → Tag 변환 함수
get_tag_from_weight() {
    weight_name=$(basename "$1")
    if [[ "$weight_name" == *"demo"* ]]; then
        echo "ade20k"
    elif [[ "$weight_name" == *"fss"* ]]; then
        echo "fss"
    else
        echo "custom"
    fi
}

# 사용할 이미지 크기 (내림차순 정렬)
#SIZES=(480 384 320 128)
SIZES=(480 384 320 256)
for IMAGE_FILE in "${IMAGES[@]}"; do
    for WEIGHT in "${WEIGHTS[@]}"; do
        TAG=$(get_tag_from_weight "$WEIGHT")
        echo "✅ Running Pytorch Model (Weight : $TAG) for image: $IMAGE_FILE with sizes: ${SIZES[*]}"
        python3 model_output.py --weights "$WEIGHT" --image "$IMAGE_FILE" --sizes ${SIZES[*]}

        echo "✅ Running TensorRT Inference (Weight : $TAG) for image: $IMAGE_FILE"
        EXE="./CPP_Project/Feature_Extractor/build/trt_feature_extractor"
        # 엔진 파일 패턴에 맞는 첫번째 파일을 선택
        ENGINES=(models/trt_engines/lseg_img_enc_vit_${TAG}__*.trt)
        if [ ${#ENGINES[@]} -eq 0 ]; then
            echo "[ERROR] No TRT engine found for tag '${TAG}'!"
            exit 1
        fi
        TRT_PATH=${ENGINES[0]}

        for SIZE in "${SIZES[@]}"; do
            echo "[INFO] Running C++ extractor: $EXE $TRT_PATH $IMAGE_FILE $SIZE"
            $EXE "$TRT_PATH" "$IMAGE_FILE" "$SIZE"
        done
    done
    echo "✅ Comparing Features for image: $IMAGE_FILE"
        python3 compare_features.py --image "$IMAGE_FILE" --sizes ${SIZES[*]}
done

echo "✅ All image comparisons completed!"