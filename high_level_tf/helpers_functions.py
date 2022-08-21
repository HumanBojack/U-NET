import numpy as np

def iou_score(y_true, y_pred):
  intersection = np.logical_and(y_pred, y_true)
  union = np.logical_or(y_pred, y_true)
  iou = np.sum(intersection) / np.sum(union)
  return iou

def dice_score(y_true, y_pred):
  intersection = np.logical_and(y_pred, y_true)
  dice = np.sum(intersection) / (np.sum(y_pred) + np.sum(y_true))
  return dice

def evaluate(y_true, y_pred):
  print('IoU        : %.2f' % (iou_score(y_true, y_pred) * 100))
  print('Dice Score : %.2f' % (dice_score(y_true, y_pred) * 100))
