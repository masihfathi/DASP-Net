import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
def find_stem(d,stem):
    for p in Path(d).rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS and p.stem==stem: return p
    return None
def read(p,size=None):
    im=Image.open(p).convert("RGB")
    if size: im=im.resize(size,Image.BICUBIC)
    return np.asarray(im).astype(np.float32)/255.0
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--title",required=True)
    ap.add_argument("--row-stems",nargs="+",required=True)
    ap.add_argument("--column",action="append",required=True,help="Label:folder")
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    cols=[x.split(":",1) for x in args.column]
    nrows=len(args.row_stems); ncols=len(cols)
    fig,axs=plt.subplots(nrows,ncols,figsize=(max(12,ncols*2.1),max(3,nrows*2.1)))
    if nrows==1: axs=np.expand_dims(axs,0)
    if ncols==1: axs=np.expand_dims(axs,1)
    size=None
    for stem in args.row_stems:
        for _,folder in cols:
            p=find_stem(folder,stem)
            if p: size=Image.open(p).convert("RGB").size; break
        if size: break
    for c,(label,_) in enumerate(cols): axs[0,c].set_title(label,fontsize=9)
    for r,stem in enumerate(args.row_stems):
        for c,(label,folder) in enumerate(cols):
            ax=axs[r,c]; p=find_stem(folder,stem)
            if not p:
                ax.text(.5,.5,"missing",ha="center",va="center"); ax.axis("off"); continue
            ax.imshow(np.clip(read(p,size),0,1)); ax.axis("off")
            if c==0: ax.set_ylabel(stem,fontsize=8)
    fig.suptitle(args.title,fontsize=13)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=300,bbox_inches="tight")
    print("Saved:",args.output)
if __name__=="__main__": main()
