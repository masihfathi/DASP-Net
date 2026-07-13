import argparse,csv,math
from pathlib import Path
import numpy as np

def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def mean(rows,k):
    vals=[]
    for r in rows:
        try:
            v=float(r[k]);
            if not math.isnan(v):vals.append(v)
        except:pass
    return float(np.mean(vals)) if vals else float("nan")

ap=argparse.ArgumentParser(); ap.add_argument("--input",action="append",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
out=[]
for item in a.input:
    label,p=item.split(":",1); rows=read(p); out.append({"method":label,"images":len(rows),"mae":mean(rows,"mae"),"psnr":mean(rows,"psnr"),"ssim":mean(rows,"ssim"),"lpips":mean(rows,"lpips")})
with Path(a.output).open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
for r in sorted(out,key=lambda x:x["psnr"],reverse=True):print(r)
