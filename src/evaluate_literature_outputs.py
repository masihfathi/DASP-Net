import argparse, csv, math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def imgs(d):
    return sorted([p for p in Path(d).rglob("*") if p.is_file() and p.suffix.lower() in EXTS])

def find_stem(d, stem):
    for p in Path(d).rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS and p.stem == stem:
            return p
    return None

def read(path, size=None):
    im=Image.open(path).convert("RGB")
    if size: im=im.resize(size, Image.BICUBIC)
    return np.asarray(im).astype(np.float32)/255.0

def mae(a,b): return float(np.mean(np.abs(a-b)))
def psnr(a,b):
    mse=float(np.mean((a-b)**2))
    return 99.0 if mse<=1e-12 else float(10*np.log10(1.0/mse))
def ssim(a,b): return float(ssim_fn(b,a,data_range=1.0,channel_axis=-1))

def lpips_model(device):
    try:
        import lpips
        m=lpips.LPIPS(net="alex").to(device); m.eval()
        print("[info] LPIPS enabled")
        return m
    except Exception as e:
        print("[warning] LPIPS disabled:", e)
        return None

def lpips_val(model,a,b,device):
    if model is None: return math.nan
    ta=torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float().to(device)*2-1
    tb=torch.from_numpy(b).permute(2,0,1).unsqueeze(0).float().to(device)*2-1
    with torch.no_grad(): v=model(ta,tb)
    return float(v.mean().detach().cpu())

def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-name",required=True)
    ap.add_argument("--low-dir",required=True)
    ap.add_argument("--high-dir",required=True)
    ap.add_argument("--method",action="append",required=True,help="Name:path")
    ap.add_argument("--output-dir",default="results/literature_exact_comparison")
    ap.add_argument("--cpu",action="store_true")
    args=ap.parse_args()

    device=torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu")))
    print("Device:",device)
    lp=lpips_model(device)

    detail=[]; summary=[]
    low_files=imgs(args.low_dir)
    for item in args.method:
        name,p=item.split(":",1)
        name=name.strip(); p=Path(p.strip())
        if not p.exists():
            print("[skip]",name,"missing:",p); continue
        rows=[]; miss=0
        for low in low_files:
            stem=low.stem
            ref=find_stem(args.high_dir,stem)
            pred=find_stem(p,stem)
            if ref is None or pred is None:
                miss+=1; continue
            size=Image.open(ref).convert("RGB").size
            r=read(ref); pr=read(pred,size=size)
            rows.append({"dataset":args.dataset_name,"method":name,"stem":stem,"prediction":str(pred),"mae":mae(pr,r),"psnr":psnr(pr,r),"ssim":ssim(pr,r),"lpips":lpips_val(lp,pr,r,device)})
        if miss: print(f"[warning] {name}: {miss} missing pairs")
        detail+=rows
        if rows:
            summary.append({"dataset":args.dataset_name,"method":name,"num_images":len(rows),
                            "mae":float(np.mean([x["mae"] for x in rows])),
                            "psnr":float(np.mean([x["psnr"] for x in rows])),
                            "ssim":float(np.mean([x["ssim"] for x in rows])),
                            "lpips":float(np.mean([x["lpips"] for x in rows if not math.isnan(x["lpips"])])) if any(not math.isnan(x["lpips"]) for x in rows) else math.nan})
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/f"{args.dataset_name.lower()}_detailed_metrics.csv",detail,["dataset","method","stem","prediction","mae","psnr","ssim","lpips"])
    write_csv(out/f"{args.dataset_name.lower()}_summary_metrics.csv",summary,["dataset","method","num_images","mae","psnr","ssim","lpips"])
    print("| Dataset | Method | Images | MAE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(summary,key=lambda x:x["psnr"],reverse=True):
        print(f"| {r['dataset']} | {r['method']} | {r['num_images']} | {r['mae']:.4f} | {r['psnr']:.4f} | {r['ssim']:.4f} | {r['lpips']:.4f} |")
if __name__=="__main__": main()
