import pandas as pd
import unicodedata
import argparse

parser = argparse.ArgumentParser(description='Combine datasets')
parser.add_argument('--root', type=str, default='offical_data/')
args = parser.parse_args()

old = pd.read_csv(args.root + 'artemis_dataset_release_v0.csv')
new = pd.read_csv(args.root + 'Contrastive.csv')

old['version'] = 'old'
new['version'] = 'new'

combined = pd.concat([new, old], axis=0)

combined['painting'] = combined['painting'].apply(lambda x: unicodedata.normalize('NFD', x))
combined['anchor_painting'] = combined['anchor_painting'].astype(str).apply(lambda x: unicodedata.normalize('NFD', x))

combined.to_csv(args.root + 'combined_artemis.csv', index=False)