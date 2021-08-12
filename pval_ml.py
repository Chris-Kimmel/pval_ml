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


# RANGE_OF_BASES_TO_INCLUDE determines how many p-values are used for the model.
# When it equals (-4, 1), the model uses 4 basepairs upstream and 1 basepair
# downstream of the putative modification site.
RANGE_OF_BASES_TO_INCLUDE = (-4, 1) # inclusive


################################################################################
############################ Imports and constants #############################
################################################################################

from os import path, makedirs, remove
from itertools import product
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
############################# Import training data #############################
################################################################################

to_concat = []
for training_pair in TRAINING_PAIR_LIST:
    # load and format each training pair separately
    nick = training_pair['model_nickname']
    ms_0b = training_pair['model_site_0b']

    d = {ms_0b + delta: delta for delta in range(r[0], r[1]+1)}
    for filepath, positive in [(training_pair['positive_dset_prs_csv'], True),
                               (training_pair['negative_dset_prs_csv'], False)]:
        # read features and labels
        to_concat.append(
            load_csv(filepath, range(ms_0b + r[0], ms_0b + r[1] + 1))
            .rename(d, axis=1, errors='raise')
            .assign(model=nick)
            .assign(positive=positive)
            .assign(prediction_site_0b=ms_0b)
            .dropna()
        )
training_dset_df = pd.concat(to_concat).set_index(['model', 'positive', 'prediction_site_0b']).dropna().sort_index()
del to_concat


################################################################################
############################## Prepare classifier ##############################
################################################################################

classifier = svm.LinearSVC(
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
#### Train on training data with k-fold cross-validation and report results ####
################################################################################

folder = model_selection.StratifiedKFold(NUMBER_OF_FOLDS, shuffle=True, random_state=855)

for model in TRAINING_PAIR_LIST:
    model_nick = model['model_nickname']
    prediction_site_0b = model['model_site_0b']

    with open(path.join(OUTPUT_DIRECTORY, 'training_results.txt'), 'at') as f:
        print(f'estimating sensitivity and specificity for {model_nick} model '
              f'using {NUMBER_OF_FOLDS}-fold cross-validation', file=f)
        print('---', file=f)

    ### convert to numpy ###

    # index levels of training_dset_df are ['model', 'positive', 'prediction_site_0b']
    pos_X = training_dset_df.loc[idx[model_nick, True, prediction_site_0b], :].to_numpy()
    neg_X = training_dset_df.loc[idx[model_nick, False, prediction_site_0b], :].to_numpy()
    n_pos = pos_X.shape[0]
    n_neg = neg_X.shape[0]
    X = np.concatenate([pos_X, neg_X], axis=0)
    y = np.concatenate([np.full((n_pos,), 1), np.full((n_neg,), 0)], axis=0)

    ### cross-validate models ###

    for train, test in folder.split(X, y):
        classifier.fit(X[train], y[train])
        report = metrics.classification_report(y[test], classifier.predict(X[test]))
        with open(path.join(OUTPUT_DIRECTORY, 'training_results.txt'), 'at') as f:
            print(report, file=f)
            print('---', file=f)


################################################################################
######################## Import datasets for evaluation ########################
################################################################################

features = pd.DataFrame( # previously named "sites"
    data=product(
        SITES_TO_EVALUATE_0B,
        range(RANGE_OF_BASES_TO_INCLUDE[0], RANGE_OF_BASES_TO_INCLUDE[1]+1),
    ),
    columns=['prediction_site_0b', 'delta'],
).assign(pos_0b=lambda x: x['prediction_site_0b'] + x['delta'])

# Each of the possible values of `delta` corresponds to a feature used by the
# model. The presence of (`pos_0b`, `prediction_site_0b`, `delta`) in the `features`
# dataframe indicates that, when the model is used to assess whether `prediction_site_0b`
# is modified, it should use the values at `pos_0b` for feature `delta`.

evaluation_dset_long = pd.concat(
    longify(load_csv(x['dset_prs_csv'])).assign(dset = x['dset_nickname'])
    for x in EVALUATION_DATASET_LIST
)
# columns: are 'read_id', 'pos_0b', 'pval', 'dset'
# no index

on_which_to_run_model = (
    evaluation_dset_long
    .merge(features, how='inner', on='pos_0b', validate='many_to_many')
    .pivot(
        index=['dset', 'read_id', 'prediction_site_0b'],
        columns='delta',
        values='pval'
    ).dropna()
)
# Signature of on_which_to_run_model:
# Index: dset, read_id, prediction_site_0b
# Columns: -4, -3, -2, -1, 0, 1


################################################################################
################### Train models using ALL the training data ###################
################################################################################

to_concat = []
for model in TRAINING_PAIR_LIST:
    model_nick = model['model_nickname']
    prediction_site_0b = model['model_site_0b']

    print(f'running {model_nick} model on evaluation datasets')

    ### convert training data to numpy ###

    # The 5-fold cross validation models each used 80% of the training data, but
    # here we use all of it.
    pos_X = training_dset_df.loc[idx[model_nick, True, prediction_site_0b], :].to_numpy()
    neg_X = training_dset_df.loc[idx[model_nick, False, prediction_site_0b], :].to_numpy()
    n_pos = pos_X.shape[0]
    n_neg = neg_X.shape[0]
    X = np.concatenate([pos_X, neg_X], axis=0)
    y = np.concatenate([np.full((n_pos,), 1), np.full((n_neg,), 0)], axis=0)

    ### train model and run on evaluation data ###

    classifier.fit(X, y)

    to_concat.append(
        pd.DataFrame(
            data=classifier.predict(on_which_to_run_model),
            index=on_which_to_run_model.index,
            columns=['predicted']
        ).assign(model=model_nick)
        .reset_index()
    )

    ### save model parameters ###

    with open(path.join(OUTPUT_DIRECTORY, 'model_parameters.txt'), 'at') as f:
        print(f'MODEL PARAMETERS FOR {model_nick}', file=f)
        print(f'classifier coefficients:\t{classifier.coef_}', file=f)
        print(f'classifier intercept:\t\t{classifier.intercept_}', file=f)
        print('---', file=f)

predictions = pd.concat(to_concat, axis=0)
del to_concat
# Columns of predictions are: dset, read_id, prediction_site_0b, model, predicted

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
