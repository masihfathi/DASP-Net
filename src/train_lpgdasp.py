from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))
from prompts import build_dasp_input
from model_lpgdasp import build_lpgdasp_net

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu: return torch.device("cpu")
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def find_by_stem(folder: Path, stem: str) -> Optional[Path]:
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS and p.stem == stem:
            return p
    return None


def to_chw_tensor(array: np.ndarray) -> torch.Tensor:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3: raise ValueError(f"Expected 3D array, got {arr.shape}")
    if arr.shape[0] in {3,4,7}: chw = arr
    elif arr.shape[-1] in {3,4,7}: chw = np.transpose(arr, (2,0,1))
    else: raise ValueError(f"Cannot infer layout from {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(chw)).float()


class LocalPairedPromptDataset(Dataset):
    def __init__(self, low_dir: str, high_dir: str, width: int = 256,
                 height: int = 256, augment: bool = False) -> None:
        self.low_dir = Path(low_dir); self.high_dir = Path(high_dir)
        self.width = width; self.height = height; self.augment = augment
        pairs = []
        for low in list_images(self.low_dir):
            high = find_by_stem(self.high_dir, low.stem)
            if high is not None: pairs.append((low, high))
        if not pairs: raise FileNotFoundError("No paired images found")
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def _read(self, p: Path) -> np.ndarray:
        im = Image.open(p).convert("RGB").resize((self.width, self.height), Image.BICUBIC)
        return np.asarray(im).astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> Dict:
        low_path, high_path = self.pairs[idx]
        low = self._read(low_path); target = self._read(high_path)
        if self.augment and random.random() < 0.5:
            low = np.ascontiguousarray(np.flip(low, 1)); target = np.ascontiguousarray(np.flip(target, 1))
        dasp = to_chw_tensor(build_dasp_input(low))
        if dasp.shape[0] != 7: raise ValueError(f"Expected 7 channels, got {dasp.shape}")
        return {"dasp_input": dasp, "target": to_chw_tensor(target), "name": low_path.stem}


def gaussian_window(size, sigma, channels, device, dtype):
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    w = (g[:,None] * g[None,:]).expand(channels,1,size,size).contiguous()
    return w


def differentiable_ssim(x, y, window_size=11, sigma=1.5):
    c = x.shape[1]; w = gaussian_window(window_size, sigma, c, x.device, x.dtype); p = window_size // 2
    ux = F.conv2d(x,w,padding=p,groups=c); uy = F.conv2d(y,w,padding=p,groups=c)
    vx = F.conv2d(x*x,w,padding=p,groups=c)-ux*ux
    vy = F.conv2d(y*y,w,padding=p,groups=c)-uy*uy
    vxy = F.conv2d(x*y,w,padding=p,groups=c)-ux*uy
    c1, c2 = 0.01**2, 0.03**2
    return (((2*ux*uy+c1)*(2*vxy+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2)+1e-8)).mean()


def gradient_loss(pred, target):
    px = pred[:,:,:,1:] - pred[:,:,:,:-1]; py = pred[:,:,1:,:] - pred[:,:,:-1,:]
    tx = target[:,:,:,1:] - target[:,:,:,:-1]; ty = target[:,:,1:,:] - target[:,:,:-1,:]
    return F.l1_loss(px,tx) + F.l1_loss(py,ty)


def total_loss(pred, target, lambda_ssim, lambda_grad):
    l1 = F.l1_loss(pred,target); ssim_loss = 1-differentiable_ssim(pred,target); grad = gradient_loss(pred,target)
    total = l1 + lambda_ssim*ssim_loss + lambda_grad*grad
    return total, {"l1":float(l1.detach().cpu()),"ssim_loss":float(ssim_loss.detach().cpu()),"grad":float(grad.detach().cpu()),"total":float(total.detach().cpu())}


def batch_metrics(pred,target):
    pred = pred.clamp(0,1); target = target.clamp(0,1)
    mae = F.l1_loss(pred,target).item(); mse = F.mse_loss(pred,target).item()
    psnr = 99.0 if mse <= 1e-12 else 10*math.log10(1/mse)
    ssim = float(differentiable_ssim(pred,target).detach().cpu())
    return {"mae":mae,"psnr":psnr,"ssim":ssim}


def mean_dict(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


def run_epoch(model, loader, device, optimizer, lambda_ssim, lambda_grad):
    training = optimizer is not None; model.train(training)
    losses=[]; metrics=[]
    for batch in loader:
        x=batch["dasp_input"].to(device); y=batch["target"].to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred=model(x).clamp(0,1); loss,comp=total_loss(pred,y,lambda_ssim,lambda_grad)
            if training:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
        losses.append(comp); metrics.append(batch_metrics(pred.detach(),y.detach()))
    out={}
    for prefix,rows in (("loss",losses),("metric",metrics)):
        for k,v in mean_dict(rows).items(): out[f"{prefix}_{k}"]=v
    return out


def save_history(history,path):
    if not history:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--adapter-mode",choices=["refine","gate","hybrid"],default="hybrid")
    ap.add_argument("--train-low-dir",required=True); ap.add_argument("--train-high-dir",required=True)
    ap.add_argument("--val-low-dir",required=True); ap.add_argument("--val-high-dir",required=True)
    ap.add_argument("--output-dir",required=True); ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch-size",type=int,default=2); ap.add_argument("--height",type=int,default=256); ap.add_argument("--width",type=int,default=256)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--weight-decay",type=float,default=0.0)
    ap.add_argument("--lambda-ssim",type=float,default=0.2); ap.add_argument("--lambda-grad",type=float,default=0.1)
    ap.add_argument("--hidden-channels",type=int,default=32); ap.add_argument("--initial-residual-scale",type=float,default=0.10)
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--num-workers",type=int,default=0); ap.add_argument("--cpu",action="store_true")
    args=ap.parse_args(); seed_everything(args.seed); device=select_device(args.cpu); print("Device:",device)
    out=Path(args.output_dir); ckpt=out/"checkpoints"; ckpt.mkdir(parents=True,exist_ok=True)
    train_ds=LocalPairedPromptDataset(args.train_low_dir,args.train_high_dir,args.width,args.height,True)
    val_ds=LocalPairedPromptDataset(args.val_low_dir,args.val_high_dir,args.width,args.height,False)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
    model=build_lpgdasp_net(args.adapter_mode,args.hidden_channels,args.initial_residual_scale).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=3)
    config={**vars(args),"device":str(device),"train_images":len(train_ds),"val_images":len(val_ds)}
    (out/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    best=-1e9; history=[]
    for epoch in range(1,args.epochs+1):
        tr=run_epoch(model,train_loader,device,opt,args.lambda_ssim,args.lambda_grad)
        va=run_epoch(model,val_loader,device,None,args.lambda_ssim,args.lambda_grad); scheduler.step(va["metric_psnr"])
        row={"epoch":epoch,"lr":opt.param_groups[0]["lr"],**{f"train_{k}":v for k,v in tr.items()},**{f"val_{k}":v for k,v in va.items()}}
        history.append(row); save_history(history,out/"history.csv")
        print(f"Epoch {epoch:03d}/{args.epochs} | train={tr['loss_total']:.4f} | val MAE={va['metric_mae']:.4f} | PSNR={va['metric_psnr']:.4f} | SSIM={va['metric_ssim']:.4f}")
        payload={"epoch":epoch,"adapter_mode":args.adapter_mode,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"metrics":va,"config":config}
        torch.save(payload,ckpt/f"lpgdasp_{args.adapter_mode}_last.pth")
        if va["metric_psnr"]>best:
            best=va["metric_psnr"]; torch.save(payload,ckpt/f"lpgdasp_{args.adapter_mode}_best.pth"); print("  saved best")
    print("Best PSNR:",best)


if __name__=="__main__": main()
