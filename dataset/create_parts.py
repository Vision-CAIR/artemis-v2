import pandas as pd
import numpy as np
import unicodedata
import shutil
import os.path as osp
import os
import pathlib

base_dir = osp.split(pathlib.Path(__file__).absolute())[0]

PATH_TO_WIKIART = '/ibex/scratch/mohameys/wiki_art_paintings/rescaled_600px_max_side/'

def func(row):
    return PATH_TO_WIKIART + row['art_style'] + '/' + row['painting'] + '.jpg'

def save_set(df, path, test=False):
    if test:
        os.makedirs(osp.split(path)[0], exist_ok=True)
        df.to_csv(path, index=False)
    else:
        os.makedirs(path, exist_ok=True)
        df.to_csv(osp.join(path, 'artemis_preprocessed.csv'), index=False)
        shutil.copy(osp.join(base_dir, 'full_combined/train', 'vocabulary.pkl'), path)
        shutil.copy(osp.join(base_dir, 'full_combined/train', 'config.json.txt'), path)


full_combined = pd.read_csv(osp.join(base_dir, 'full_combined/train/artemis_preprocessed.csv'))
full_combined['painting'] = full_combined['painting'].apply(lambda x: unicodedata.normalize('NFD', x))
full_combined['art_style'] = full_combined['art_style'].apply(lambda x: unicodedata.normalize('NFD', x))
full_combined['image_file'] = full_combined.apply(func, axis=1)
full_combined['image_file'] = full_combined['image_file'].apply(lambda x: unicodedata.normalize('NFD', x))
full_combined['grounding_emotion'] = full_combined['emotion']

new_set = full_combined[full_combined['version']=='new']
old_full_set = full_combined[full_combined['version']=='old']
# create combined training set
old_small_set = old_full_set.sample(len(new_set), random_state=42)
old_large_set = old_full_set.sample(min(2*len(new_set), len(old_full_set)), random_state=42)
combined_set = pd.concat((new_set, old_small_set), axis=0)

# save training
save_set(df=old_full_set, path=osp.join(base_dir, 'old_full/train'))
save_set(df=new_set, path=osp.join(base_dir, 'new/train'))
save_set(df=combined_set, path=osp.join(base_dir, 'combined/train'))
save_set(df=old_small_set, path=osp.join(base_dir, 'old_small/train'))
save_set(df=old_large_set, path=osp.join(base_dir, 'old_large/train'))

# create all four test datasets
new_test = new_set[new_set['split'] == 'test']
old_all_test = old_full_set[old_full_set['split'] == 'test']
unique_ids = old_all_test.art_style + old_all_test.painting
unique_ids = set(unique_ids.unique())
np.random.seed(42)
sub_ids = np.random.choice(list(unique_ids), int(len(unique_ids)//2), replace=False)
old_test = old_all_test[(old_all_test.art_style+old_all_test.painting).isin(sub_ids)]

combined_test = pd.concat((new_test, old_test), axis=0)
a40_test = old_full_set[old_full_set['split']=='rest'].copy()
a40_test.loc[:,'split'] = 'test'

save_set(df=new_test, path=osp.join(base_dir, 'test_new/test_new.csv'), test=True)
save_set(df=combined_test, path=osp.join(base_dir, 'test_all/test_all.csv'), test=True)
save_set(df=old_test, path=osp.join(base_dir, 'test_old/test_old.csv'), test=True)
save_set(df=old_all_test, path=osp.join(base_dir, 'test_ref/test_ref.csv'), test=True)
save_set(df=a40_test, path=osp.join(base_dir, 'test_40/test_40.csv'), test=True)
