'''
pval_ml.py

Chris Kimmel
kimmel.95@osu.edu (school)
chris.kimmel@live.com (personal)
'''
# pylint: disable=wrong-import-position,line-too-long,invalid-name


################################################################################
########################### User-customizable parts ############################
################################################################################

OUTPUT_DIRECTORY = './pval_ml_output'


TRAINING_PAIR_LIST = [
    {'positive_dset_prs_csv': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/single_m6A_pattern/082720_8079m6A56bp_oligo10x_carrierRNA_ligation/tombo_fishers0/082720_8079m6A_positive_fishers0.csv',
     'negative_dset_prs_csv': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/single_m6A_pattern/082720_8079m6A56bp_oligo10x_carrierRNA_ligation/tombo_fishers0/082720_8079m6A_negative_fishers0.csv',
     'model_site_0b': 8078,
     'model_nickname': '8079_synthetic',
    },
    {'positive_dset_prs_csv': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/single_m6A_pattern/8975m6A/121520_m6A8975_splintDNA_oligo5x/8975_pos_fisher0.csv',
     'negative_dset_prs_csv': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/single_m6A_pattern/8975m6A/121520_m6A8975_splintDNA_oligo5x/8975_neg_fisher0.csv',
     'model_site_0b': 8974,
     'model_nickname': '8975_synthetic',
    },
    {'positive_dset_prs_csv': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/single_m6A_pattern/8989m6A/121520_m6A8988_splintDNA_oligo5x/8989_pos_fishers0.csv',
     'negative_dset_prs_csv': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/single_m6A_pattern/8989m6A/121520_m6A8988_splintDNA_oligo5x/8989_neg_fishers0.csv',
     'model_site_0b': 8988,
     'model_nickname': '8989_synthetic',
    },
]


EVALUATION_DATASET_LIST = [
    {'dset_prs_csv': '/fs/project/PAS1405/kimmel/data/fast5s/Native/Native_fishers0.csv',
     'dset_nickname': 'Trizol',
    },
    {'dset_prs_csv': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/WT_cellular/23456_WT_cellular_fishers0.csv',
     'dset_nickname': 'WT_cellular',
    },
    {'dset_prs_csv': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/ctrl_data_set/data_Olivier/IVT_fishers0.csv',
     'dset_nickname': 'IVT',
    },
    {'dset_prs_csv': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/FTO_KO_cellular/FTO_KO_cellular_fishers0.csv',
     'dset_nickname': 'FTO_KO',
    },
    {'dset_prs_csv': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/ALKBH5_KO_cellular/061821_ALKBH5_KO_293T_Cellular_RNA/061821_ALKBH5_KO_cellular_len8K_fishers0.csv',
     'dset_nickname': 'ALKBH5_KO_cellular_8K',
    },
]

'''
SITES_TO_EVALUATE_0B = [
    8033, # 7
    8078, # 1
    8109, # 8
    8716, # 2
    8974, # 3
    8988, # 4.1
    8995, # 4.2
    9096, # 6
    9113, # 10
    9119, # 6
    9130, # 9
]
'''
SITES_TO_EVALUATE_0B = list(range(0, 10**4))

# RANGE_OF_BASES_TO_INCLUDE determines how many p-values are used for the model.
# When it equals (-4, 1), the model uses 4 basepairs upstream and 1 basepair
# downstream of the putative modification site.
RANGE_OF_BASES_TO_INCLUDE = (-4, 1) # inclusive


################################################################################
############################ Imports and constants #############################
################################################################################

from os import path, makedirs, remove
from itertools import product
from Typing import Tuple
import errno

import numpy as np
import pandas as pd
idx = pd.IndexSlice

from sklearn import model_selection, svm, metrics


NUMBER_OF_FOLDS = 5


################################################################################
############################# Helpful subroutines ##############################
################################################################################

def debyte_str(s):
    '''If s looks like "b'abc123xy'", return "abc123xy". Otherwise return s.'''
    return s[2:-1] if s[0:2] == "b'" and s[-1] == "'" else s


def load_csv(filepath, poss=None):
    '''Load per-read stats from a CSV file into a Pandas DataFrame'''
    usecols = None if poss is None else ['read_id'] + [str(x) for x in poss]
    retval = (
        pd.read_csv(filepath, header=0, index_col='read_id', usecols=usecols)
        .rename_axis('pos_0b', axis=1)
    )
    retval.columns = retval.columns.astype(int)
    retval.index = [debyte_str(s) for s in retval.index]
    retval = retval.rename_axis('read_id')
    return retval


def longify(df):
    '''Convert dataframe output of load_csv to a long format'''
    return df.stack().rename('pval').reset_index()


class dataset():
    '''
    An object of this class stands for the p-values of a single p-value CSV
    file. (Such files are produced by prsconv2.py and prsconv3.)
    '''

    def __init__(self, filepath):
        '''
        Here filepath points to a per-read statistics file as output by prsconv2
        or prsconv3
        '''
        self._data = load_csv(filepath)

        # Fill in gaps in missing positions
        present_cols = self._data.columns
        min_pos, max_pos = min(present_cols), max(present_cols)
        missing_poss = set(range(min_pos, max_pos+1)) - set(self._data.columns) 
        missing_pos_df = pd.DataFrame({p: np.nan} for p in missing_poss)
        self._data = pd.concat([self._data, missing_pos_df], axis=1) # sorted automatically

    def apply_model(self, linear_svc: smv.LinearSVC, r: Tuple[int]) -> pd.DataFrame:
        '''
        Apply a linear support-vector machine to the dataset.

        Arguments:
            linear_svc:
                A trained support-vector machine from sklearn. It is understood
                that linear_svc is trained to determine whether a site is
                methylated by using the p-values in its vicinity.
            r:
                A tuple of integers like (-4, 1) describing what the features
                of linear_svc are. The left entry says how many bases upstream
                are used by the model, and the right entry says how many bases
                downstream.
                    For instance, if r=(-4, 1), then to call the methylation
                of nucleotide #7005, p-values at nucleotides #7001 through
                #7006 are used.

        Returns:
            self._data, but with 0s, 1s, and NaNs representing methylation calls
            at every site in the dataset
        '''
        assert len(linear_svc._coef) == r[1] - r[0] + 1, f'invalid r: {r}'
        assert r[1] > r[0], f'invalid r: {r}'

        v = sliding_window_view(self._data, r[1] - r[0] + 1, axis=1)
        predictions = ((v @ linear_svc._coef.ravel()) > linear_svc._thresh) \
            .reshape(self._data.shape)

        return pd.DataFrame(predictions, columns=self._data.columns,
            index=self._data.index)

    def restrict_columns(self, lo_incl: int, hi_incl: int) -> None:
        '''
        Return the same dataframe, but only with columns lo_incl, lo_incl+1, ...
        hi_incl-1, hi_incl and no others.
        '''
        return self._data.loc[:, lo_incl:hi_incl] # loc uses inclusive indexing

    def to_numpy(self) -> np.ndarray:
        '''
        Wrapper for pd.DataFrame.to_numpy()
        '''
        return self._data.to_numpy()

    def dropna(self) -> None:
        '''
        Wrapper for pd.DataFrame.dropna()
        '''
        self._data = self._data.dropna()

def get_model():
    '''
    Return an svm.LinearSVC model with parameter choices appropriate for our
    purposes.
    '''
    return svm.LinearSVC(
        penalty='l2', # default
        loss='squared_hinge', # default
        dual=True, # default
        tol=0.0001, # default
        C=1.0, # default
        fit_intercept=True, # default
        intercept_scaling=1.0, # default
        class_weight='balanced', # NOT default
        random_state=855, # NOT default
        max_iter=1000, # default
    )


################################################################################
#################################### Setup #####################################
################################################################################

r = RANGE_OF_BASES_TO_INCLUDE

makedirs(OUTPUT_DIRECTORY, exist_ok = True)

for filename in ['predictions.csv', 'model_parameters.txt', 'training_results.txt', 'predictions_summary.csv']:
    # copied from https://stackoverflow.com/a/10840586
    try:
        remove(path.join(OUTPUT_DIRECTORY, filename))
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise e


################################################################################
################ Estimate model accuracy using cross-validation ################
################################################################################

for training_pair in TRAINING_PAIR_LIST:
    locals().update(training_pair)
    # above line loads:
    #   positive_dset_prs_csv, negative_dset_prs_csv, model_site_0b, model_nickname

    with open(path.join(OUTPUT_DIRECTORY, 'training_results.txt'), 'at') as f:
        print(f'estimating sensitivity and specificity for {model_nick} model '
              f'using {NUMBER_OF_FOLDS}-fold cross-validation', file=f)
        print('See https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report')
        print('---', file=f)

    pos_X, neg_X = (
        dataset(filepath).restrict_columns(*r).dropna().to_numpy()
        for filepath in positive_dset_prs_csv, negative_dset_prs_csv
    )
    n_pos = pos_X.shape[0]
    n_neg = neg_X.shape[0]
    y = np.concatenate([np.full((n_pos,), 1),
                        np.full((n_neg,), 0)], axis=0)
    X = np.concatenate([pos_X, neg_X], axis=0)

    for train, test in folder.split(X, y):
        classifier = get_model().fit(X[train], y[train])
        report = metrics.classification_report(y[test], classifier.predict(X[test]))
        with open(path.join(OUTPUT_DIRECTORY, 'training_results.txt'), 'at') as f:
            print(report, file=f)
            print('---', file=f)

    del pos_X, neg_X, X, y # free up some memory


################################################################################
#################### Train models on full training datasets ####################
################################################################################

classifiers_and_descriptions = []
for training_pair in TRAINING_PAIR_LIST:
    locals().update(training_pair)
    # above line loads:
    #   positive_dset_prs_csv, negative_dset_prs_csv, model_site_0b, model_nickname

    with open(path.join(OUTPUT_DIRECTORY, 'training_results.txt'), 'at') as f:
        print(f'estimating sensitivity and specificity for {model_nick} model '
              f'using {NUMBER_OF_FOLDS}-fold cross-validation', file=f)
        print('See https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report')
        print('---', file=f)

    pos_X, neg_X = (
        dataset(filepath).restrict_columns(*r).dropna().to_numpy()
        for filepath in positive_dset_prs_csv, negative_dset_prs_csv
    )
    n_pos = pos_X.shape[0]
    n_neg = neg_X.shape[0]
    y = np.concatenate([np.full((n_pos,), 1),
                        np.full((n_neg,), 0)], axis=0)
    X = np.concatenate([pos_X, neg_X], axis=0)

    classifier = get_model().fit(X, y)
    classifiers_and_descriptions.append(classifier, description)

    with open(path.join(OUTPUT_DIRECTORY, 'model_parameters.txt'), 'at') as f:
        print(f'MODEL PARAMETERS FOR {model_nick}', file=f)
        print(f'classifier coefficients:\t{classifier.coef_}', file=f)
        print(f'classifier intercept:\t\t{classifier.intercept_}', file=f)
        print('---', file=f)

    del pos_X, neg_X, X, y # free up some memory


################################################################################
###################### Run models on evaluation datasets #######################
################################################################################

for evaluation_dataset in EVALUATION_DATASET_LIST:
    locals().update(evaluation_dataset)
    # above line loads:
    #   dset_prs_csv, dset_nickname

    this_dataset = dataset(filepath)

    for classifer, model in classifiers_and_descriptions:
        locals().update(training_pair)
        # above line loads:
        #   positive_dset_prs_csv, negative_dset_prs_csv, model_site_0b, model_nickname
        lo_incl, hi_incl = (model_site_0b + x for x in r)
        restricted = this_dataset.restrict(lo_incl, hi_incl)
        results = restricted.apply_model(classifier)


(
    predictions
    .merge(on_which_to_run_model.reset_index(), how='inner', on=['dset', 'read_id', 'prediction_site_0b'], validate='many_to_one')
    .loc[:, ['model', 'dset', 'read_id', 'prediction_site_0b', 'predicted'] + list(range(RANGE_OF_BASES_TO_INCLUDE[0], RANGE_OF_BASES_TO_INCLUDE[1] + 1))]
    .to_csv(path.join(OUTPUT_DIRECTORY, 'predictions.csv'), index=False)
)

(
    predictions
    .groupby(['model', 'dset', 'prediction_site_0b'])
    .agg(fraction_predicted_methylated = ('predicted', 'mean'))
    .reset_index()
    .loc[:, ['model', 'dset', 'prediction_site_0b', 'fraction_predicted_methylated']]
    .to_csv(path.join(OUTPUT_DIRECTORY, 'predictions_summary.csv'), index=False)
)

print('done')
