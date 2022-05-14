
def calculate_metrics(preds, labels):
  tp = 0
  tn = 0
  fp = 0
  fn = 0

  #print (preds, labels)

  for i in range(len(preds)):
    if preds[i] == 1 and labels[i] == 1:
      tp = tp + 1
    elif preds[i] == 0 and labels[i] == 0:
      tn = tn + 1
    elif preds[i] == 1 and labels[i] == 0:
      fp = fp + 1
    else: # preds[i] == 0 and labels[i] == 1:
      fn = fn + 1

  precision = 0 #tp / (tp + fp)
  recall = 0 #tp / (tp + fn)
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  f1 = 0 #2 * precision * recall / (precision + recall)
  return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'presision': precision, 'recall': recall, 'accuracy': accuracy, 'f1': f1}


if __name__ == '__main__':
  labels = [1,0,1,1,1,0,0,0,1,1,1,1]
  predictions = [0,0,0,1,1,0,0,0,0,0,1,0]

  print(adjust_predictions(predictions, labels))
