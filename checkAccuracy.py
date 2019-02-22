import numpy as np 
import matplotlib.pyplot as pyplot

def dice(pred, gt):
    eps = 1e-7
    intersection = (pred*gt)
    dsc = (intersection.sum().sum() + eps) / (pred.sum().sum() + gt.sum().sum() + eps)
    return dsc

# whatever the name of the prediction image is (hardcoded for now)
pred = np.load('prediction.jpg')
# whatever the name of the groundtruth image is (hardcoded as well)
gt = np.load('groundtruth.jpg')
dice_score = dice(pred, gt)
print("accuracy is:" np.round(dice_score,decimals=3))