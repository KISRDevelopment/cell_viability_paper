import numpy as np 
import glob 
import glob

from scipy import interp
import sklearn.metrics

def collate_results(cv_dir):

    files = glob.glob("%s/*" % cv_dir)

    baccs = []
    per_class_bacc = []
    per_class_roc = []
    order = []
    for file in files:
        d = np.load(file, allow_pickle=True)

        r = d['r'].item()

        preds = d['preds']
        y_target = d['y_target']

        baccs.append(r['bacc'])
        per_class_bacc.append(r['per_class_baccs'])
        per_class_roc.append(r['per_class_auc_roc'])
        order.append((d['rep'].item(), d['fold'].item()))

    return baccs, per_class_bacc, per_class_roc, order

def average_cm(cv_dir):

    files = glob.glob("%s/*" % cv_dir)

    cms = []
    for file in files:
        d = np.load(file, allow_pickle=True)

        if 'cm' in d:
            cm = d['cm']
        else:
            preds = d['preds']
            y_target = d['y_target']
            _, cm = eval_classifier(y_target, preds)
        
        cm = cm / np.sum(cm, axis=1, keepdims=True)

        if cm.shape[0] != 2:
            print(cm)
            
        cms.append(cm)
    cms = np.array(cms)

    return np.mean(cms, axis=0)

def compute_stars(pvalue, alpha):
    """ compute the stars to visualize a p-value """
    PVALUE_STARS = [
        1.,
        1/5.,
        1/50.,
        1/500.
    ]

    for i in range(len(PVALUE_STARS)-1, -1, -1):
        level = PVALUE_STARS[i] * alpha
        if pvalue < level:
            return i+1

    return 0


# Based on
# https://stackoverflow.com/questions/51442818/average-roc-curve-across-folds-for-multi-class-classification-case-in-sklearn
def average_roc_curve(cv_dir, klass):

    BASE_FPR = np.linspace(0, 1, 101)

    files = glob.glob("%s/*" % cv_dir)

    tprs = []
    for file in files:
        d = np.load(file, allow_pickle=True)

        preds = d['preds'][:,klass]
        y_target = d['y_target'] == klass

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_target, preds)
        tpr = interp(BASE_FPR, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    return BASE_FPR, np.mean(tprs, axis=0)

def eval_classifier(y_target, preds):

    y_pred = np.argmax(preds, axis=1)

    cm = sklearn.metrics.confusion_matrix(y_target, y_pred)

    overall_acc = sklearn.metrics.accuracy_score(y_target, y_pred)

    overall_bacc = sklearn.metrics.balanced_accuracy_score(y_target, y_pred)

    avg_f1 = sklearn.metrics.f1_score(y_target, y_pred, average='macro')

    avg_pres = sklearn.metrics.precision_score(y_target, y_pred, average='macro')

    avg_recall = sklearn.metrics.recall_score(y_target, y_pred, average='macro')

    log_prob = np.sum(np.log(preds[np.arange(preds.shape[0]), y_target]))

    num_labels = preds.shape[1]

    # per label metrics
    aucs = []
    prs = []
    f1s = []
    precisions = []
    recalls = []
    baccs = []
    for b in range(num_labels):
        y_bin = y_target == b
        target_pred = y_pred == b

        if np.sum(y_bin) == 0:
            aucs.append(0)
            prs.append(0)
            f1s.append(0)
            precisions.append(0)
            recalls.append(0)
            baccs.append(0)
            continue 
        
        auc = sklearn.metrics.roc_auc_score(y_bin, preds[:,b])
        pr = sklearn.metrics.average_precision_score(y_bin, preds[:,b])

        f1 = sklearn.metrics.f1_score(y_bin, target_pred)
        precision = sklearn.metrics.precision_score(y_bin, target_pred)
        recall = sklearn.metrics.recall_score(y_bin, target_pred)
        bacc = sklearn.metrics.balanced_accuracy_score(y_bin, target_pred)

        aucs.append(auc)
        prs.append(pr)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        baccs.append(bacc)


    avg_roc = np.mean(aucs)
    avg_pr = np.mean(prs)

    return {
        "acc" : overall_acc,
        "bacc" : overall_bacc,
        "f1" :  avg_f1, 
        "pres" : avg_pres, 
        "recall" : avg_recall,
        "auc_roc" : avg_roc,
        "auc_pr" : avg_pr,
        "log_prob" : log_prob,
        "per_class_f1" : f1s,
        "per_class_pres" : precisions,
        "per_class_recall" : recalls,
        "per_class_baccs" : baccs,
        "per_class_pr" : prs,
        "per_class_auc_roc" : aucs,
        "per_class_auc_pr" : prs
    }, cm

