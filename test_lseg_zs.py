import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.lseg_module_zs import LSegModuleZS
from fewshot_data.common.logger import Logger, AverageMeter
from fewshot_data.common.vis import Visualizer
from fewshot_data.common.evaluation import Evaluator
from fewshot_data.common import utils
from fewshot_data.data.dataset import FSSDataset
from modules.lseg_full import LSegFull   


# ---------------------------------------------------------------------
#  공통 베이스 - MultiEvalModule 이 요구하는 6개 속성을 자동 주입
# ---------------------------------------------------------------------
class AdapterBase(torch.nn.Module):
    def _inherit(self, lseg_full):                # ← ★ 함수 이름을 _inherit 로
        cfg = lseg_full.module if hasattr(lseg_full, "module") else lseg_full
        self.base_size   = cfg.base_size
        self.crop_size   = cfg.crop_size
        self.mean        = getattr(cfg, "mean", [0.5, 0.5, 0.5])
        self.std         = getattr(cfg,  "std",  [0.5, 0.5, 0.5])
        # _up_kwargs 혹은 up_kwargs 둘 중 존재하는 이름을 안전하게 가져옴
        self._up_kwargs  = getattr(cfg, "_up_kwargs",
                                   getattr(cfg, "up_kwargs",
                                           {'mode':'bilinear','align_corners':True}))
        self.evaluate = self.forward        # MultiEvalModule 이 호출

# ---------------------------------------------------------------------


# ---------------------
# 1) PyTorch 어댑터
# ---------------------
from modules.models.lseg_vit import forward_vit
class PTWrapper(torch.nn.Module):
    """
    LSegModule 을 체크포인트에서 로드해서,
    .net(img) 호출만으로 (1,512,hf,wf) 특징 맵을 뽑아내는 래퍼
    """
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        super().__init__()
        # checkpoint 로드
        self.module: LSegModule = LSegModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            map_location=device,
            backbone="clip_vitl16_384",
            num_features=256,
            crop_size=480,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
            readout="ignore",
        )
        self.module.net.to(device).eval()
        self.device = device

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (1,3,H,W) 텐서 (정규화된 상태)
        → module.net(img) 호출해서 (1,512,hf,wf) 특징 맵 리턴
        """
        img = img.to(self.device)
        with torch.no_grad():
            feat = self.module.net(img)   # head1까지 포함된 512채널 특징
        return feat


class PTAdapter(AdapterBase):
    """
    * flip-ensemble → encoder 특징 추출
    * head-0(conv 1×1) + head-1(CLIP text 유사도) → softmax 후 가중합
    * 최종 class-probability 맵을 리턴  (B,C,H,W)
    """
    def __init__(self, wrapper: PTWrapper, full: LSegFull):
        super().__init__()
        self.wrapper     = wrapper                  # encoder

        self._inherit(full)

        self.txt_feat_t  = full.txt_feat_t.float()  # (Ncls,512)
        self.logit_scale = full.logit_scale.float()
        self.up_kwargs   = self._up_kwargs          # 동일 참조

        self.head0 = getattr(wrapper.module.net.scratch, "output_conv", None)
        self.has_head0 = isinstance(self.head0, torch.nn.Conv2d)

    # ――― 내부: flip ensemble 후 (B,512,hf,wf)
    def _encode(self, img):
        f1 = self.wrapper(img)
        f2 = torch.flip(self.wrapper(torch.flip(img,[-1])), [-1])
        return (f1+f2)*0.5

    def forward(self, img):
        feat = self._encode(img)                   # (B,512,hf,wf)

        # ── head-1 (CLIP text cos-sim) ───────────
        B,Cenc,hf,wf = feat.shape
        f_norm = F.normalize(feat, dim=1)\
                   .permute(0,2,3,1).reshape(-1,Cenc)          # (B*hf*wf,512)
        sim = torch.matmul(f_norm, self.txt_feat_t.t())        # (Npix,Ncls)
        logits1 = (sim * self.logit_scale)\
                    .view(B,hf,wf,-1).permute(0,3,1,2)         # (B,Ccls,hf,wf)
        logits1 = F.interpolate(logits1, size=img.shape[-2:], **self.up_kwargs)
        prob1 = torch.softmax(logits1, 1)

        # ── head-0 : LSeg 모듈의 classify_head0 사용
        if self.has_head0:
            logits0 = self.head0(feat)
            logits0 = F.interpolate(logits0, img.shape[-2:], **self._up_kwargs)
            prob0   = torch.softmax(logits0, 1)
            prob    = prob0 + 0.2 * prob1
        else:
            prob = 0.2 * prob1              # (B,Ncls,hf,wf)
        
        return prob
    
# ---------------------
# 2) TRT 래퍼 & 어댑터
# ---------------------
class TRTWrapper:
    def __init__(self, engine_path):
        logger  = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            buf = f.read()
        self.engine  = runtime.deserialize_cuda_engine(buf)
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()
        
        # ── 바인딩 파악 (입력 1, 출력 ≥1)
        self.in_name  = self.engine.get_tensor_name(0)
        self.out_names = [self.engine.get_tensor_name(i)
                          for i in range(1, self.engine.num_io_tensors)]

        # 채널 수로 어떤 출력이 feature / pred0 인지 구분
        feat_name = pred_name = None
        for n in self.out_names:
            _, C, _, _ = self.engine.get_tensor_shape(n)
            if C == 512:   feat_name = n
            elif C == 150: pred_name = n
        assert feat_name and pred_name, "엔진에 512ch / 150ch 출력이 모두 필요합니다"
        self.feat_name, self.pred_name = feat_name, pred_name

        # 버퍼 핸들 저장 {name: Ptr}
        self.d_ptr = {}
        self.buf_cap = {}
    
    def _malloc(self, name, nbytes):
         if name in self.buf_cap and self.buf_cap[name] >= nbytes:
             return
         if name in self.d_ptr: self.d_ptr[name].free()
         self.d_ptr[name] = cuda.mem_alloc(nbytes)
         self.buf_cap[name] = nbytes

    def infer(self, x: torch.Tensor):
        arr = x.cpu().numpy().astype(np.float32)
        b,c,h,w = arr.shape
        self.context.set_input_shape(self.in_name, (b,c,h,w))
        # 출력 shape (두 개)
        b2, C2, H2, W2 = self.context.get_tensor_shape(self.pred_name)
        b3, C3, H3, W3 = self.context.get_tensor_shape(self.feat_name)
        # 버퍼 할당
        self._malloc(self.in_name , arr.nbytes)
        self._malloc(self.pred_name, b2*C2*H2*W2*4)
        self._malloc(self.feat_name, b3*C3*H3*W3*4)
        # 주소 바인딩
        self.context.set_tensor_address(self.in_name, int(self.d_in))
        self.context.set_tensor_address(self.pred_name, int(self.d_ptr[self.pred_name]))
        self.context.set_tensor_address(self.feat_name, int(self.d_ptr[self.feat_name]))
        # 입력 복사 + 실행
        cuda.memcpy_htod_async(self.d_in, arr, self.stream)
        self.context.execute_async_v3(self.stream.handle)

        # 결과 복사
        pred = np.empty((b2,C2,H2,W2), dtype=np.float32)
        feat = np.empty((b3,C3,H3,W3), dtype=np.float32)
        cuda.memcpy_dtoh_async(pred, self.d_ptr[self.pred_name], self.stream)
        cuda.memcpy_dtoh_async(feat, self.d_ptr[self.feat_name], self.stream)
        self.stream.synchronize()
        return pred, feat

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument(
            "--model", type=str, default="encnet", help="model name (default: encnet)"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="resnet50",
            help="backbone name (default: resnet50)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="ade20k",
            help="dataset name (default: pascal12)",
        )
        parser.add_argument(
            "--workers", type=int, default=16, metavar="N", help="dataloader threads"
        )
        parser.add_argument(
            "--base-size", type=int, default=520, help="base image size"
        )
        parser.add_argument(
            "--crop-size", type=int, default=480, help="crop image size"
        )
        parser.add_argument(
            "--train-split",
            type=str,
            default="train",
            help="dataset train split (default: train)",
        )
        # training hyper params
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            testing (default: same as batch size)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        # checking point
        parser.add_argument(
            "--weights", type=str, default=None, help="checkpoint to test"
        )
        # evaluation option
        parser.add_argument(
            "--eval", action="store_true", default=False, help="evaluating mIoU"
        )

        parser.add_argument(
            "--acc-bn",
            action="store_true",
            default=False,
            help="Re-accumulate BN statistics",
        )
        parser.add_argument(
            "--test-val",
            action="store_true",
            default=False,
            help="generate masks on val set",
        )
        parser.add_argument(
            "--no-val",
            action="store_true",
            default=False,
            help="skip validation during training",
        )

        parser.add_argument(
            "--module",
            default='',
            help="select model definition",
        )

        # test option
        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )

        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )

        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )

        parser.add_argument(
            "--jobname",
            type=str,
            default="default",
            help="select which dataset",
        )

        parser.add_argument(
            "--no-strict",
            dest="strict",
            default=True,
            action="store_false",
            help="no-strict copy the model",
        )

        parser.add_argument(
            "--use_pretrained",
            type=str,
            default="True",
            help="whether use the default model to intialize the model",
        )

        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )

        # fewshot options
        parser.add_argument(
            '--nshot', 
            type=int, 
            default=1
            )
        parser.add_argument(
            '--fold', 
            type=int, 
            default=0, 
            choices=[0, 1, 2, 3]
            )
        parser.add_argument(
            '--nworker', 
            type=int, 
            default=0
            )
        parser.add_argument(
            '--bsz', 
            type=int, 
            default=1
            )
        parser.add_argument(
            '--benchmark', 
            type=str, 
            default='pascal',
            choices=['pascal', 'coco', 'fss', 'c2p']
            )
        parser.add_argument(
            '--datapath', 
            type=str, 
            default='fewshot_data/Datasets_HSN'
            )

        parser.add_argument(
            "--activation",
            choices=['relu', 'lrelu', 'tanh'],
            default="relu",
            help="use which activation to activate the block",
        )

        parser.add_argument('--trt-engine', type=str, default='', help='TensorRT engine file path')


        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args


def test(args):
    module_def = LSegModuleZS

    module = module_def.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path=args.datapath,
        dataset=args.dataset,
        backbone=args.backbone,
        aux=args.aux,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=args.ignore_index,
        dropout=0.0,
        scale_inv=args.scale_inv,
        augment=False,
        no_batchnorm=False,
        widehead=args.widehead,
        widehead_hr=args.widehead_hr,
        map_locatin="cpu",
        arch_option=args.arch_option,
        use_pretrained=args.use_pretrained,
        strict=args.strict,
        logpath='fewshot/logpath_4T/',
        fold=args.fold,
        block_depth=0,
        nshot=args.nshot,
        finetune_mode=False,
        activation=args.activation,
    )

    Evaluator.initialize()
    if args.backbone in ["clip_resnet101"]:
        FSSDataset.initialize(img_size=480, datapath=args.datapath, use_original_imgsize=False, imagenet_norm=True)
    else:
        FSSDataset.initialize(img_size=480, datapath=args.datapath, use_original_imgsize=False)
    # dataloader
    args.benchmark = args.dataset
    dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    model = module.net.eval().cuda()
    # model = module.net.model.cpu()

    print(model)

    scales = (
        [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        if args.dataset == "citys"
        else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )  

    f = open("logs/fewshot/log_fewshot-test_nshot{}_{}.txt".format(args.nshot, args.dataset), "a+")

    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        image = batch['query_img']
        target = batch['query_mask']
        class_info = batch['class_id']
        # pred_mask = evaluator.parallel_forward(image, class_info)
        pred_mask = model(image, class_info)
        # assert pred_mask.argmax(dim=1).size() == batch['query_mask'].size()
        # 2. Evaluate prediction
        if args.benchmark == 'pascal' and batch['query_ignore_idx'] is not None:
            query_ignore_idx = batch['query_ignore_idx']
            area_inter, area_union = Evaluator.classify_prediction(pred_mask.argmax(dim=1), target, query_ignore_idx)
        else:
            area_inter, area_union = Evaluator.classify_prediction(pred_mask.argmax(dim=1), target)

        average_meter.update(area_inter, area_union, class_info, loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    # Write evaluation results
    average_meter.write_result('Test', 0)
    test_miou, test_fb_iou = average_meter.compute_iou()

    Logger.info('Fold %d, %d-shot ==> mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, args.nshot, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
    f.write('{}\n'.format(args.weights))
    f.write('Fold %d, %d-shot ==> mIoU: %5.2f \t FB-IoU: %5.2f\n' % (args.fold, args.nshot, test_miou.item(), test_fb_iou.item()))
    f.close()
                


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    test(args)
