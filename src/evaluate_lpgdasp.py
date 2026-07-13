from __future__ import annotations
import argparse,csv,math,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
sys.path.append(str(Path(__file__).resolve().parent))
from model_lpgdasp import build_lpgdasp_net
from train_lpgdasp import LocalPairedPromptDataset,batch_metrics,select_device

def load_lpips(device):
    try:
        import lpips
        m=lpips.LPIPS(net="alex").to(device); m.eval(); return m
    except Exception as e:
        print("[warning] LPIPS unavailable:",e); return None

def save_image(t,path):
    a=t.clamp(0,1).detach().cpu().permute(1,2,0).numpy(); Image.fromarray((a*255).round().astype(np.uint8)).save(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--adapter-mode",choices=["refine","gate","hybrid"],required=True)
    ap.add_argument("--low-dir",required=True); ap.add_argument("--high-dir",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--output-csv",required=True)
    ap.add_argument("--height",type=int,default=256); ap.add_argument("--width",type=int,default=256); ap.add_argument("--batch-size",type=int,default=1)
    ap.add_argument("--hidden-channels",type=int,default=32); ap.add_argument("--initial-residual-scale",type=float,default=0.10); ap.add_argument("--cpu",action="store_true")
    args=ap.parse_args(); device=select_device(args.cpu); print("Device:",device)
    ds=LocalPairedPromptDataset(args.low_dir,args.high_dir,args.width,args.height,False); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False)
    model=build_lpgdasp_net(args.adapter_mode,args.hidden_channels,args.initial_residual_scale).to(device)
    ck=torch.load(args.checkpoint,map_location=device); model.load_state_dict(ck.get("model_state_dict",ck)); model.eval(); lp=load_lpips(device)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); rows=[]
    with torch.no_grad():
        for batch in loader:
            x=batch["dasp_input"].to(device); y=batch["target"].to(device); pred=model(x).clamp(0,1)
            for i,name in enumerate(batch["name"]):
                m=batch_metrics(pred[i:i+1],y[i:i+1]); l=float("nan")
                if lp is not None:
                    l=float(lp(pred[i:i+1]*2-1,y[i:i+1]*2-1).mean().detach().cpu())
                rows.append({"name":name,"adapter_mode":args.adapter_mode,**m,"lpips":l}); save_image(pred[i],out/f"{name}.png")
    with Path(args.output_csv).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    def mean(k):
        vals=[float(r[k]) for r in rows if not math.isnan(float(r[k]))]; return float(np.mean(vals)) if vals else float("nan")
    print(f"{args.adapter_mode}: MAE={mean('mae'):.4f}, PSNR={mean('psnr'):.4f}, SSIM={mean('ssim'):.4f}, LPIPS={mean('lpips'):.4f}")

if __name__=="__main__": main()
