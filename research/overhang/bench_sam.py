"""Split SAM 2 time per tray into preprocessing (CPU), GPU model (encoder + decoder) and
post-processing (CPU), in fp32 and bf16. Run on any CUDA machine to compare GPUs:
  python overhang/bench_sam.py RUN_DIR [N]
RUN_DIR/cache/NNN.jpg are half-res frames (as in the pipeline)."""
import glob, os, sys, time
import cv2, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam_tray import load, dev

run = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
files = sorted(glob.glob(f"{run}/cache/*.jpg"))[:N]
proc, model = load()
print("device", torch.cuda.get_device_name(0) if dev == "cuda" else dev, " torch", torch.__version__)
imgs = [Image.fromarray(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)) for f in files]
pts = [[[200, 150], [500, 150], [200, 400], [500, 400], [364, 272]]]; lbl = [[1, 1, 1, 1, 0]]
sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)

for dtype in (torch.float32, torch.bfloat16):
    m = model.to(dtype)
    tp = tg = tq = 0.0
    with torch.no_grad():
        for i, img in enumerate([imgs[0]] * 5 + imgs):          # 5 warm-up passes
            a = time.perf_counter()
            inp = proc(images=img, input_points=[pts], input_labels=[lbl], return_tensors="pt").to(dev)
            inp["pixel_values"] = inp["pixel_values"].to(dtype)
            sync(); b = time.perf_counter()
            o = m(**inp, multimask_output=True)
            sync(); c = time.perf_counter()
            mk = proc.post_process_masks(o.pred_masks.float().cpu(), inp["original_sizes"])[0][0]
            _ = o.iou_scores.float().cpu()
            d = time.perf_counter()
            if i >= 5:
                tp += b - a; tg += c - b; tq += d - c
    n = len(imgs)
    print(f"{str(dtype):15s} prep {tp/n*1000:6.1f} ms   GPU model {tg/n*1000:6.1f} ms   post {tq/n*1000:6.1f} ms   "
          f"total {(tp+tg+tq)/n*1000:6.1f} ms/tray   peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GB" if dev == "cuda" else "")
