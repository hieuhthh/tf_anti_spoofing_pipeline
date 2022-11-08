import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
from glob import glob

def preprocess_img_video(frame, img_size):
    frame = frame[:, :, [2, 1, 0]]
    frame = cv2.resize(frame, (img_size)) 
    return frame

def load_video(path, im_size, max_frames):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = preprocess_img_video(frame, (im_size, im_size))
            frames.append(frame)
    finally:
        cap.release()
        
    frames = np.array(frames)
    
    if len(frames) >= max_frames:
        frames = frames[:max_frames]
    else:
        pad = np.zeros((max_frames-len(frames), im_size, im_size, 3))
        frames = np.concatenate((frames, pad))
    
    return frames