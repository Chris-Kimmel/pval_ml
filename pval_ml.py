'''
pval_ml.py

Chris Kimmel
kimmel.95@osu.edu (school)
chris.kimmel@live.com (personal)
'''
# pylint: disable=wrong-import-position,line-too-long,invalid-name


### User-customizable parts

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


SITES_TO_EVALUATE_ZB = [
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


### Imports and constants

from os import path, makedirs, remove
from itertools import product
import errno

import numpy as np
import pandas as pd
idx = pd.IndexSlice

from sklearn import model_selection, svm, metrics


NUMBER_OF_FOLDS = 5


### Helpful subroutines

def debyte_str(s):
    '''If s looks like "b'abc123xy'", return "abc123xy". Otherwise return s.'''
    return s[2:-1] if s[0:2] == "b'" and s[-1] == "'" else s


def load_csv(filepath, poss):
    '''Load per-read stats from a CSV file into a Pandas DataFrame'''
    retval = (
        pd.read_csv(filepath, header=0, index_col=0, usecols=map(str, poss))
        .rename_axis('pos_0b', axis=1)
    )
    retval.columns = retval.columns.astype(int)
    retval.index = [debyte(s) for s in retval.index]
    retval = retval.rename_axis('read_id')
    return retval


def longify(df):
    '''Convert dataframe output of load_csv to a long format'''
    return df.stack().rename('pval').reset_index()


### Setup

r = RANGE_OF_BASES_TO_INCLUDE

makedirs(OUTPUT_DIRECTORY, exist_ok = True)

for filename in ['predictions.csv', 'model_parameters.txt', 'training_results.txt', 'predictions_summary.csv']:
    # copied from https://stackoverflow.com/a/10840586
    try:
        remove(path.join(OUTPUT_DIRECTORY, filename))
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


### Import training data

to_concat = []
for training_pair in TRAINING_PAIR_LIST:
    nick = training_pair['model_nickname']
    ms_0b = training_pair['model_site_0b']
    lo = ms_0b + r[0]
    hi = ms_0b + r[1]
    d = {ms_0b + delta: delta for delta in range(lo, hi+1)}
    for filepath, positive in [(training_pair['positive_dset_prs_csv'], True),
                               (training_pair['negative_dset_prs_csv'], False)]:
        # read data and create useful columns
        to_concat.append(
            load_csv(filepath, range(lo, hi+1))
            .rename(d, axis=1)
            .assign(model = nick)
            .assign(positive = positive)
            .assign(site_0b: ms_0b)
        )
training_dset_df = pd.concat(to_concat)
del to_concat


### prepare classifier ###

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


### Train on training data with k-fold cross-validation and report results

folder = model_selection.StratifiedKFold(NUMBER_OF_FOLDS, shuffle=True, random_state=855)

for model_nick, site_0b in [(model['model_nickname'], model['model_site_0b']) for model in TRAINING_PAIR_LIST]:

    print(f'estimating sensitivity and specificity for {model_nick} model '
          f'using {NUMBER_OF_FOLDS}-fold cross-validation')
    print('---')

    ### convert to numpy ###

    pos_X = training_dset_df.loc[idx[model_nick, True, :, site_0b], :].to_numpy()
    n_pos = pos_X.shape[0]
    neg_X = training_dset_df.loc[idx[model_nick, False, :, site_0b], :].to_numpy()
    n_neg = neg_X.shape[0]
    # pos on top of neg!
    X = np.concatenate([pos_X, neg_X], axis=0)
    y = np.concatenate([np.full((n_pos,), 1), np.full((n_neg,), 0)], axis=0)

    ### cross-validate models ###

    for train, test in folder.split(X, y):
        classifier.fit(X[train], y[train])
        report = metrics.classification_report(y[test], classifier.predict(X[test]))
        print(report, file=f)
        print('---')


### Import datasets for evaluation

sites = pd.DataFrame(
    data=product(
        SITES_TO_EVALUATE_ZB,
        range(RANGE_OF_BASES_TO_INCLUDE[0], RANGE_OF_BASES_TO_INCLUDE[1]+1),
    ),
    columns=['site_0b', 'delta'],
).assign(pos_0b=lambda x: x['site_0b'] + x['delta'])

to_concat = []
for evaluation_dataset in EVALUATION_DATASET_LIST:
    # load dataset
    df = (
        load_csv(evaluation_dataset['dset_prs_csv'])
        .assign(dset = evaluation_dataset['dset_nickname'])
    )
    to_concat.append(df)
    del df
evaluation_dset_long = pd.concat(to_concat)
del to_concat

# TODO: following lines were written when evaluation_dset_long was in long form, but now it is in wide form
# TODO: change variable name of evaluation_dset_long
evaluation_sites = (
    pd.merge(
        evaluation_dset_long,
        sites,
        on='pos_0b', how='inner', validate='many_to_one'
    ).pivot(
        index=['dset', 'read_id', 'site_0b'],
        columns='delta',
        values='pval'
    ).dropna()
)
del evaluation_dset_long
# Signature of evaluation_sites:
# Index: dset, read_id, site_0b
# Columns: -4, -3, -2, -1, 0, 1

### Train models using ALL the training data

to_concat = []
for model_nick, site_0b in [(model['model_nickname'], model['model_site_0b']) for model in TRAINING_PAIR_LIST]:

    print(f'running {model_nick} model on evaluation datasets')

    ### convert to numpy ###

    pos_X = training_dset_df.loc[idx[model_nick, True, :, site_0b], :].to_numpy()
    n_pos = pos_X.shape[0]
    neg_X = training_dset_df.loc[idx[model_nick, False, :, site_0b], :].to_numpy()
    n_neg = neg_X.shape[0]
    # pos on top of neg!
    X = np.concatenate([pos_X, neg_X], axis=0)
    y = np.concatenate([np.full((n_pos,), 1), np.full((n_neg,), 0)], axis=0)

    ### train model ###

    classifier.fit(X, y)
    to_concat.append(
        pd.DataFrame(
            data=classifier.predict(evaluation_sites),
            index=evaluation_sites.index,
            columns=['predicted']
        ).assign(model=model_nick)
    )

    with open(path.join(OUTPUT_DIRECTORY, 'model_parameters.txt'), 'at') as f:
        print(f'MODEL PARAMETERS FOR {model_nick}', file=f)
        print(f'classifier coefficients:\t{classifier.coef_}', file=f)
        print(f'classifier intercept:\t\t{classifier.intercept_}', file=f)
        print('---', file=f)
predictions = pd.concat(to_concat, axis=0) #.reset_index().rename({'site_0b': 'prediction_site_0b'}, axis=1)
del to_concat
# Signature of predictions:
# Index: dset, read_id, site_0b
# Columns: model, predicted
# print(predictions.columns) # TODO: delete line (debugging)

(
    predictions
    .reset_index()
    .merge(evaluation_sites.reset_index(), how='inner', on=['dset', 'read_id', 'site_0b'], validate='many_to_one')
    .assign(prediction_site_1b = lambda x: x['site_0b'] + 1)
    .loc[:, ['model', 'dset', 'read_id', 'prediction_site_1b' , 'predicted'] + list(range(RANGE_OF_BASES_TO_INCLUDE[0], RANGE_OF_BASES_TO_INCLUDE[1] + 1))]
    .to_csv(path.join(OUTPUT_DIRECTORY, 'predictions.csv'), index=False)
)

(
    predictions
    .reset_index()
    .assign(prediction_site_1b = lambda x: x['site_0b'] + 1)
    .groupby(['model', 'dset', 'prediction_site_1b'])
    .agg(fraction_predicted_methylated = ('predicted', 'mean'))
    .reset_index()
    .loc[:, ['model', 'dset', 'prediction_site_1b', 'fraction_predicted_methylated']]
    .to_csv(path.join(OUTPUT_DIRECTORY, 'predictions_summary.csv'), index=False)
)

print('done')
