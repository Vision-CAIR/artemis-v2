"""
Various simple (basic) functions in the "utilities".

The MIT License (MIT)
Originally created at 8/31/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

from numpy.lib.arraysetops import unique
from scipy.sparse import data
import torch
import multiprocessing as mp
import dask.dataframe as dd
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import pdb

def iterate_in_chunks(l, n):
    """Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def df_parallel_column_apply(df, func, column_name):
    n_partitions = mp.cpu_count() * 4
    d_data = dd.from_pandas(df, npartitions=n_partitions)

    res =\
    d_data.map_partitions(lambda df: df.apply((lambda row: func(row[column_name])), axis=1))\
    .compute(scheduler='processes')

    return res


def cross_entropy(pred, soft_targets):
    """ pred: unscaled logits
        soft_targets: target-distributions (i.e., sum to 1)
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))


def make_train_test_val_splits(dataset_df, loads, random_seed, unique_id_column=None):
    """ Split the data into train/val/test.
    :param dataset_df: pandas Dataframe containing the dataset (e.g., ArtEmis)
    :param loads: list with the three floats summing to one for train/val/test
    :param random_seed: int
    :return: changes the dataset_df in-place to include a column ("split") indicating the split of each row
    """
    if sum(loads) != 1:
        raise ValueError()

    train_size, val_size, test_size = loads
    print("Using a {},{},{} for train/val/test purposes".format(train_size, val_size, test_size))
    
    test_num = int((test_size+val_size) * len(dataset_df[dataset_df.version == 'new']))
    dataset_df['unique_id'] = dataset_df['art_style']+dataset_df['painting']
    gdf = dataset_df.groupby(by=['unique_id', 'version']).agg(
        {
            'repetition': lambda x: x.unique(),
        }
    ).reset_index()
    start_rep = 5
    test_ids = []
    while (test_num > 0) and (start_rep < 10):
        gdf_rep = gdf[(gdf.version=='new') & (gdf.repetition==start_rep)]
        curr_cnt = len(gdf_rep) * start_rep
        if test_num > curr_cnt:
            test_ids.extend(gdf_rep.unique_id.values.flatten())
            test_num -= curr_cnt
            start_rep += 1
        else:
            test_ids.extend(gdf_rep.unique_id.values.flatten()[:int(test_num//start_rep)])
            test_num -= curr_cnt
            start_rep += 1
    print(f'New test set has maximum repetition {start_rep-1}')

    new_set_unique_ids = set(gdf[gdf.version=='new'].unique_id.values)
    new_set_train_ids = np.array(list(new_set_unique_ids - set(test_ids)))
    test_end = int(len(test_ids)*(test_size/(test_size+val_size)))
    new_set_test_ids = np.array(test_ids)[:test_end]
    new_set_val_ids = np.array(test_ids)[test_end:]

    df = dataset_df
    ## unique id
    # if unique_id_column is None:
    unique_id = df.unique_id # default for ArtEmis
    anchor_id = df.anchor_art_style + df.anchor_painting
    anchor_id = anchor_id.dropna()
    # else:
    #     unique_id = df[unique_id_column]

    unique_ids = set(unique_id.unique())
    # unique_ids.sort()
    unique_ids -= new_set_unique_ids

    anchor_ids = set(anchor_id.unique())
    # anchor_ids.sort()
    anchor_ids -= new_set_unique_ids

    train_len = len(unique_ids) * train_size
    test_len = len(unique_ids) * (test_size + val_size)

    rem_train = train_len - len(anchor_ids)

    # assert rem_train < 0, 'unique anchors is more than remaining images'
    while rem_train < 0:
        warnings.warn('Anchor paintings are more than remaining paintings .... Removing some anchor painting')
        anchor_ids.pop()
        rem_train += 1
    # unique_ids_rem = np.array([i for i in unique_ids if i not in anchor_ids])
    unique_ids_rem = np.array(list(unique_ids - anchor_ids))

    train, rest = train_test_split(unique_ids_rem, test_size=(test_len)/(test_len+rem_train), random_state=random_seed)

    train = set(np.concatenate((train, list(anchor_ids), new_set_train_ids)))

    if val_size != 0:
        val, test = train_test_split(rest, test_size=test_size/(test_size+val_size), random_state=random_seed)
    else:
        test = rest
    test = set(np.concatenate((test, new_set_test_ids)))
    assert len(test.intersection(train)) == 0

    def mark_example(x):
        if x in train:
            return 'train'
        elif x in test:
            return 'test'
        else:
            return 'val'

    df = df.assign(split=unique_id.apply(mark_example))
    df.drop(columns=['anchor_art_style', 'anchor_painting', 'unique_id'], inplace=True)
    return df