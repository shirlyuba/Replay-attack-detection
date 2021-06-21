import numpy as np


eps = 20

def A_feature(img1, img2):
    return np.abs(img1 - img2)

def A_feature_map(imgs):
    diff = np.array([])
    for i in range(len(imgs)-1):
        diff = np.append(diff, A_feature(imgs[i], imgs[i+1]))
    diff = diff.reshape(shape)
    return diff

def F_feature(img1, img2):
    return (np.abs(img1 - img2) > eps).astype(int) * 255

def F_feature_map(imgs):
    diff = np.array([])
    for i in range(len(imgs)-1):
        diff = np.append(diff, F_feature(imgs[i], imgs[i+1]))
    diff = diff.reshape(shape)
    return diff


