import utils.eval_funcs
import sklearn.metrics
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors 

def average_results(cv_dir, thresholds):

    files = utils.eval_funcs.get_files(cv_dir)

    cms_by_t = { t: [] for t in thresholds }
    for file in files:
        print(file)
        d = np.load(file, allow_pickle=True)

        r = d['r'].item()
        y_target = d['y_target']

        for t in thresholds:
            y_pred = d['preds'][:,1] > t
            cm = sklearn.metrics.confusion_matrix(y_target, y_pred)
            cm = cm / np.sum(cm, axis=1, keepdims=True)
            cms_by_t[t].append(cm)

      
    return { t: np.mean(cms_by_t[t], axis=0) for t in thresholds }

r = average_results('../results/task_yeast_gi_hybrid_binary/mn', 1-np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]))
#r = average_results('../results/task_human_gi/mn', 1-np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]))

np.set_printoptions(precision=2)
for t, cm in r.items():
    print('%0.2f' % (1-t))
    print(cm)
    print("bacc: %0.2f" % (np.mean(np.diag(cm))))
    print("fdr: %0.2f, tpr: %0.2f" % (cm[1, 0], cm[0,0]))
    print()