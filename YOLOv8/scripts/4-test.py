import os
import sys
from ultralytics import YOLO

def main(argv):
    if len(argv) < 2:
        print("Usage: python 4-test.py [0-shot/1-shot/3-shot/6-shot/9-shot]")
        return
    name = argv[1]
    if name not in ('0-shot', '1-shot', '3-shot', '6-shot', '9-shot'):
        print("Usage: python 4-test.py [0-shot/1-shot/3-shot/6-shot/9-shot]")
        return
    
    dir_path = f'../data/saved_models/{name}' 

    avg_f1, avg_p, avg_r = 0, 0, 0
    for m in os.listdir(dir_path):
        model_path = dir_path + '/' + m
        model = YOLO(model_path)
        
        tp, tn, fp, fn = 0, 0, 0, 0
        for type in ('positive', 'negative'):
            results = model(f'../data/ijmond/n-shot/{name}/test/{type}/', verbose=False)

            for result in results:
                y_pred = result.probs.top1
                if type == 'positive' and y_pred == 1:
                    tp += 1
                elif type == 'positive' and y_pred == 0:
                    fn += 1
                elif type == 'negative' and y_pred == 1:
                    fp += 1
                else:
                    tn += 1
        try:
            p = tp / (tp + fp)
        except ZeroDivisionError:
            p = 0
        try:
            r = tp / (tp + fn)
        except ZeroDivisionError:
            r = 0
        try:
            f1 = 2 * ((p * r) / (p + r))
        except ZeroDivisionError:
            f1 = 0

        print(f'Model {m} results:')
        print(f'Precision: {p}')
        print(f'Recall: {r}')
        print(f'F1-score: {f1}\n')
        
        avg_p += p
        avg_r += r
        avg_f1 += f1
    print(avg_p / 6)
    print(avg_r / 6)
    print(avg_f1 / 6)

if __name__ == "__main__":
    main(sys.argv)