import pandas as pd
import unicodedata

old = pd.read_csv('official_data/artemis_dataset_release_v0.csv')
new = pd.read_csv('official_data/Contrastive.csv')

old['version'] = 'old'
new['version'] = 'new'

combined = pd.concat([new, old], axis=0)

combined['painting'] = combined['painting'].apply(lambda x: unicodedata.normalize('NFD', x))
combined['anchor_painting'] = combined['anchor_painting'].astype(str).apply(lambda x: unicodedata.normalize('NFD', x))

combined.to_csv('official_data/combined_artemis.csv', index=False)