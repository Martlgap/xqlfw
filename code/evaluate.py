import argparse
import numpy as np
import pandas as pd
from os import makedirs
from os.path import dirname, join
from glob import glob
from termcolor import cprint
from tqdm import tqdm
from tabulate import tabulate
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import paired_distances
# This file contains modified code from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch


parser = argparse.ArgumentParser()
parser.add_argument("--source_embeddings", required=True, type=str, default=None,
                    help="Path to a folder with *.npy files containing facial embeddings and pairwise labels.")
parser.add_argument("--csv", type=bool, default=True,
                    help="Shows results as csv instead of a table.")
parser.add_argument("--save", type=bool, default=True,
                    help="Saves the results as .npy file in <source_embeddings>/results.")
config = parser.parse_args()


def k_fold_cross_validation(embs: np.array, labels: np.array, metric: str = "cosine", k: int = 10):
    """ Perform k-fold cross validation
    :param embs: embeddings (feature vectors) of pairs in consecutively a list [emb0a, emb0b, emb_1a, emb_1b, ...]
    :param labels: list with booleans for embeddings pairs [True, False, ...] len(labels) = len(embeddings) / 2
    :param metric: which metric to use for distance between embeddings, i.e. 'cosine' or 'euclidean', ...
    :param k: number of folds to use
    :return: for each fold -> true positive rates, false positive rates, accuracies, thresholds
    """

    def _evaluate(_thresh: float, _dists: np.array, _labels: np.array):
        """ Evaluate TP, FP, TN, and FN -> calculate accuracy
        :param _thresh:
        :param _dists: array of distances
        :param _labels:
        :return:
        """
        predictions = np.less(_dists, _thresh)
        tp = np.sum(np.logical_and(predictions, _labels))
        fp = np.sum(np.logical_and(predictions, np.logical_not(_labels)))
        tn = np.sum(np.logical_and(np.logical_not(predictions), np.logical_not(_labels)))
        fn = np.sum(np.logical_and(np.logical_not(predictions), _labels))
        actual_tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        actual_fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / _dists.size
        return actual_tpr, actual_fpr, acc

    embs_1 = embs[0::2]
    embs_2 = embs[1::2]

    nrof_pairs = len(labels)
    thresholds = np.arange(0, 2, 0.001)
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=k, shuffle=False)

    tprs = np.zeros((k, nrof_thresholds))
    fprs = np.zeros((k, nrof_thresholds))
    accuracies = np.zeros(k)
    best_thresholds = np.zeros(k)
    indices = np.arange(nrof_pairs)

    dists = paired_distances(embs_1, embs_2, metric=metric)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold_train in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = _evaluate(threshold_train, dists[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)

        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold_test in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = _evaluate(
                threshold_test, dists[test_set], labels[test_set]
            )

        _, _, accuracies[fold_idx] = _evaluate(
            thresholds[best_threshold_index], dists[test_set], labels[test_set]
        )

    return tprs, fprs, accuracies, best_thresholds


def main(cfg):
    files = glob(cfg.source_embeddings + "*.npy")
    cprint("Source Location: {} | found {} files.".format(cfg.source_embeddings, len(files)), "green")

    df = pd.DataFrame([])  # Using pandas dataframe to store all results

    for file in tqdm(files, desc="Evaluating"):
        embeddings, labels = np.load(file, allow_pickle=True)
        tprs, fprs, acc, thresh = k_fold_cross_validation(embeddings, labels)
        results = {"tprs": [np.mean(tprs, 0)],
                   "fprs": [np.mean(fprs, 0)],
                   "acc": acc.mean(),
                   "thresh": thresh.mean()}
        df = df.append(pd.DataFrame(results, index=[file.split("/")[-1]]))

        if cfg.save:  # Save the results
            makedirs(join(dirname(file), "results"), exist_ok=True)
            np.save(join(dirname(file), "results", file.split("/")[-1]), results)

    cprint("Results: ", "green")
    if cfg.csv:  # Print the results as comma separated values
        print(df.drop(columns=["tprs", "fprs"]).to_csv())
    else:  # Print the results as a nice table
        print(tabulate(df.drop(columns=["tprs", "fprs"]), headers="keys", floatfmt=".4f", tablefmt="github"))

if __name__ == "__main__":
    main(config)
