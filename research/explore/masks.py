import cv2, numpy as np, sys
for p in sys.argv[1:]:
    f=cv2.imread(p); hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    red=(((h<12)|(h>160))&(s>70)&(v>60)).astype(np.uint8)*255
    yel=((h>18)&(h<40)&(s>100)&(v>120)).astype(np.uint8)*255
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32)
    k=(9,9); m=cv2.blur(g,k); sd=np.sqrt(np.maximum(cv2.blur(g*g,k)-m*m,0))
    smooth=((sd<9)&(v<110)).astype(np.uint8)*255
    vis=np.zeros_like(f); vis[...,2]=red; vis[...,1]=yel; vis[...,0]=smooth
    cv2.imwrite(p.replace('.jpg','_mask.png'),np.hstack([f,vis]))
    # print hsv of tray region sample
