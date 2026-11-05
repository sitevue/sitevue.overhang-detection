import cv2, numpy as np, sys
cap=cv2.VideoCapture(sys.argv[1]); i=0; rows=[]
while True:
    ok,f=cap.read()
    if not ok: break
    if i%2==0:
        s=cv2.resize(f,(364,272),interpolation=cv2.INTER_AREA)
        hsv=cv2.cvtColor(s,cv2.COLOR_BGR2HSV)
        h,sat,v=hsv[...,0],hsv[...,1],hsv[...,2]
        red=((h<10)|(h>165))&(sat>80)&(v>50)
        rows.append((i, red.mean()))
    i+=1
np.save("explore/redfrac.npy",np.array(rows))
r=np.array(rows); on=r[:,1]>0.03
print("frames",i,"pct meat",on.mean().round(3))
for k in range(0,len(r),50): print(f"{r[k,0]/10:6.1f}s {r[k,1]:.3f} "+"#"*int(r[k,1]*100))
