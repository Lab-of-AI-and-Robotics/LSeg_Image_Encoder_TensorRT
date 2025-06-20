import torch
import os, sys
import glob
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
# 프로젝트 루트 경로를 import 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.lseg_module import LSegModule
from modules.lseg_module_zs import LSegModuleZS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, size):
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return tf(img).unsqueeze(0).to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_dir", type=str, default="models/weights",
        help="Root folder containing ViT/ and Resnet/ subfolders with .ckpt files"
    )
    parser.add_argument(
        "--backbones", nargs="+",
        choices=["vit","resnet"], default=["vit","resnet"],
        help="Which backbone(s) to run (default: both)"
    )
    parser.add_argument(
        "--images", nargs="+", required=True,
        help="List of input image files"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[480,384,320,256],
        help="List of image sizes"
    )
    args = parser.parse_args()

    base_output = "outputs/feature_maps/pt"
    for backbone in args.backbones:
        ckpt_dir = os.path.join(args.weights_dir, "ViT" if backbone=="vit" else "Resnet")
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        print(f"[INFO] Backbone={backbone} → {len(ckpts)} checkpoints")
        for ckpt in ckpts:
            weight_name = os.path.splitext(os.path.basename(ckpt))[0]
            print(f"  ▶ Checkpoint={weight_name}.ckpt")

            # 모델 로드
            if backbone=="vit":
                module = LSegModule.load_from_checkpoint(
                    checkpoint_path=ckpt, map_location="cpu",
                    backbone="clip_vitl16_384", aux=False, num_features=256,
                    crop_size=max(args.sizes), readout="project",
                    aux_weight=0, se_loss=False, se_weight=0,
                    ignore_index=255, dropout=0.0, scale_inv=False,
                    augment=False, no_batchnorm=False,
                    widehead=True, widehead_hr=False,
                    arch_option=0, block_depth=0, activation="lrelu"
                ).net.to(device)
            else:
                module = LSegModuleZS.load_from_checkpoint(
                    checkpoint_path=ckpt, map_location="cpu",
                    data_path="data/", dataset="ade20k",
                    backbone="clip_resnet101", aux=False, num_features=256,
                    aux_weight=0, se_loss=False, se_weight=0,
                    base_lr=0, batch_size=1, max_epochs=0,
                    ignore_index=255, dropout=0.0, scale_inv=False,
                    augment=False, no_batchnorm=False,
                    widehead=False, widehead_hr=False,
                    arch_option=0, use_pretrained="True", strict=False,
                    logpath="fewshot/logpath_4T/", fold=0,
                    block_depth=0, nshot=1, finetune_mode=False,
                    activation="lrelu"
                ).net.to(device)
            module.eval()

            # 각 이미지 × 사이즈별로 feature 저장
            for img_path in args.images:
                img_stem = os.path.splitext(os.path.basename(img_path))[0]
                print(f"    • Image={img_stem}")
                for size in args.sizes:
                    inp = load_image(img_path, size)
                    with torch.no_grad():
                        feat = module(inp).cpu().numpy()

                    out_dir = os.path.join(base_output, backbone, weight_name, img_stem)
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, f"{size}.npy")
                    np.save(out_file, feat)
                    print(f"      ✔ Saved → {out_file}")

    print("[INFO] All PT feature maps generated.")

if __name__ == "__main__":
    main()