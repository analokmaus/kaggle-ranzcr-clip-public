import random
from collections import Counter, defaultdict
import gc

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.utils import check_random_state


class MultiStratifiedGroupKFold:

    def __init__(self, n_splits: int = 5, random_state: int = 0):
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def multi_label_stratified_group_k_fold(label_arr: np.array, gid_arr: np.array, n_fold: int, seed: int = 42):
        """
        create multi-label stratified group kfold indexs.

        reference: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
        input:
            label_arr: numpy.ndarray, shape = (n_train, n_class)
                multi-label for each sample's index using multi-hot vectors
            gid_arr: numpy.array, shape = (n_train,)
                group id for each sample's index
            n_fold: int. number of fold.
            seed: random seed.
        output:
            yield indexs array list for each fold's train and validation.
        """
        np.random.seed(seed)
        random.seed(seed)
        # start_time = time.time()
        n_train, n_class = label_arr.shape
        gid_unique = sorted(set(gid_arr))
        n_group = len(gid_unique)

        # # aid_arr: (n_train,), indicates alternative id for group id.
        # # generally, group ids are not 0-index and continuous or not integer.
        gid2aid = dict(zip(gid_unique, range(n_group)))
    #     aid2gid = dict(zip(range(n_group), gid_unique))
        aid_arr = np.vectorize(lambda x: gid2aid[x])(gid_arr)

        # # count labels by class
        cnts_by_class = label_arr.sum(axis=0)  # (n_class, )

        # # count labels by group id.
        col, row = np.array(sorted(enumerate(aid_arr), key=lambda x: x[1])).T
        cnts_by_group = coo_matrix(
            (np.ones(len(label_arr)), (row, col))
        ).dot(coo_matrix(label_arr)).toarray().astype(int)
        del col
        del row
        cnts_by_fold = np.zeros((n_fold, n_class), int)

        groups_by_fold = [[] for fid in range(n_fold)]
        # pair of aid and cnt by group
        group_and_cnts = list(enumerate(cnts_by_group))
        np.random.shuffle(group_and_cnts)
        # print("finished preparation", time.time() - start_time)
        for aid, cnt_by_g in sorted(group_and_cnts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for fid in range(n_fold):
                # # eval assignment.
                cnts_by_fold[fid] += cnt_by_g
                fold_eval = (cnts_by_fold / cnts_by_class).std(axis=0).mean()
                cnts_by_fold[fid] -= cnt_by_g

                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = fid

            cnts_by_fold[best_fold] += cnt_by_g
            groups_by_fold[best_fold].append(aid)
        # print("finished assignment.", time.time() - start_time)

        gc.collect()
        idx_arr = np.arange(n_train)
        for fid in range(n_fold):
            val_groups = groups_by_fold[fid]

            val_indexs_bool = np.isin(aid_arr, val_groups)
            train_indexs = idx_arr[~val_indexs_bool]
            val_indexs = idx_arr[val_indexs_bool]

            # print("[fold {}]".format(fid), end=" ")
            # print("n_group: (train, val) = ({}, {})".format(
            #     n_group - len(val_groups), len(val_groups)), end=" ")
            # print("n_sample: (train, val) = ({}, {})".format(
            #     len(train_indexs), len(val_indexs)))

            yield train_indexs, val_indexs

    def split(self, X, y, groups):
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if isinstance(groups, (pd.DataFrame, pd.Series)):
            groups = groups.values
        return self.multi_label_stratified_group_k_fold(
            y, groups, self.n_splits, self.random_state)
