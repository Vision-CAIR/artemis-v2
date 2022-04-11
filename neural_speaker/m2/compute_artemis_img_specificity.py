import pandas as pd
import sys 
import nltk
import pickle

nltk.download('wordnet')

from scipy import nanmean
from scipy.io import loadmat
from scipy.stats import spearmanr
from reimplementation import analyze_corpus, image_specificity

def load_images():
    df = pd.read_csv('/home/haydark/artemis.csv')
    df['img'] = df['art_style'] + '/' + df['painting']
    df = df.groupby('img')['utterance'].apply(list).reset_index(name='utterances')
    return df


df = load_images()

images = df['utterances'].values
vectorizer, analyzer = analyze_corpus(images)
scores = []

for i, image in enumerate(images):
    if i % 10000 == 0:
        print(i)
    score = image_specificity(image, vectorizer, analyzer)
    scores.append(score)

with open('artemis_specifity_scores.pickle', 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

df['specifity'] = scores
df.to_csv("artemi_specifity.csv",index=False)