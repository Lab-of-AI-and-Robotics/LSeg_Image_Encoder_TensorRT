import torch
from modules.lseg_module import LSegModule
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--image",  type=str,      required=True)
parser.add_argument("--sizes",  type=int, nargs='+', required=True)
parser.add_argument("--weights",type=str, default="models/weights/ViT/demo_e200.ckpt")
args = parser.parse_args()

os.makedirs("outputs", exist_ok=True)

def load_image(path, size):
    img = Image.open(path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return tf(img).unsqueeze(0).to(device)

checkpoint_path = args.weights
tag = ("ade20k" if "demo" in checkpoint_path or "ade20k" in checkpoint_path
       else "fss" if "fss" in checkpoint_path else "custom")

for size in args.sizes:
    # 1) 모델 로드 & eval
    model = LSegModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone="clip_vitl16_384",
        aux=False,
        crop_size=size,
        num_features=256,
        readout="project",
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        ignore_index=255,
        dropout=0.0,
        scale_inv=False,
        augment=False,
        no_batchnorm=False,
        widehead=True,
        widehead_hr=False,
        map_location=device,
        arch_option=0,
        block_depth=0,
        activation="lrelu"
    ).net.to(device)
    model.eval()

    # 2) 입력 로드 & 추론
    inp = load_image(args.image, size)
    with torch.no_grad():
        out = model(inp)

    # 3) 저장
    out_np = out.cpu().numpy()
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_path = f"outputs/pt_vit_{tag}_{size}_fMap_{stem}.npy"
    np.save(out_path, out_np)
    print(f"[INFO] Saved PyTorch feature map → {out_path}")

print("[INFO] All feature maps saved!")
