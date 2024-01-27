from insightface.app import FaceAnalysis
import numpy as np
import argparse
import cv2
ap=argparse.ArgumentParser()
ap.add_argument("--image1",type=str,help="Path of the first image")
ap.add_argument("--image2",type=str,help="Path of the second image")
args=ap.parse_args
image1_p=args.image1
image2_p=args.image2
fa=FaceAnalysis(providers=['CPUEecutionProvider'],name="buffalo_l")
fa.prepare(ctx_id=0,det_size=(640,640))
image1=cv2.imread(image1_p)
image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
result1=fa.get(image1)


image2=cv2.imread(image2_p)
image2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
result2=fa.get(image2)

distance=np.sprt(np.sum((result1[0]["embedding"]-result2[0]["embedding"])**2))
threshold=25
if distance<threshold:
    print("Same person")
else:
    print("Diffrent persons")