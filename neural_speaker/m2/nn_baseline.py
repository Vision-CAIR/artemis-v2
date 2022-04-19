import argparse
import pickle
import random
import unicodedata

import numpy as np
import pandas as pd

import evaluation
from evaluation.bleu import Bleu

import pathlib
import os.path as osp

ROOT_DIR = osp.split(pathlib.Path(__file__).parent.parent.absolute())[0]


random.seed(1234)
np.random.seed(1234)

def load_image_splits(csv_file, test_csv):
    splits = pd.read_csv(csv_file)
    splits = splits[splits['split']!='test']
    test_splits = pd.read_csv(test_csv)
    splits = pd.concat([splits,test_splits], axis=0).dropna(axis=1)
    splits['art_paint'] = '/' + splits[['art_style', 'painting']].agg('/'.join, axis=1)
    images_df = splits.groupby('art_paint')['utterance'].apply(list).reset_index(name='utterances')
    train_paints = splits[splits['split']=='train']['art_paint'].unique()
    test_paints = splits[splits['split']=='test']['art_paint'].unique()
    
    return images_df, train_paints, test_paints

def pick_captions(caption_set, mode='easy'):
    if mode == 'easy':
        return [np.random.choice(caption_set)]
    else:
        lex_sim_scores = []
        for i, c in enumerate(caption_set):
            caps = {'gen':{}, 'gts':{}}
            caps['gen'][i] = [c]
            caps['gts'][i] = [c_star for c_star in caption_set if c != c_star]
            gts = evaluation.PTBTokenizer.tokenize(caps['gts'])
            gen = evaluation.PTBTokenizer.tokenize(caps['gen'])
            
            metric = Bleu()
            score, _ = metric.compute_score(gts, gen)
            lex_sim_scores.append(np.mean(score))
        # pick caption with the highest lexical similarity
        return [caption_set[np.argmax(lex_sim_scores)]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN retrieve Baseline')
    parser.add_argument('-splits_file', type=str)
    parser.add_argument('-test_splits_file', type=str)
    parser.add_argument('--nn_file', type=str, default='vgg_nearest_neighbors.pkl')
    parser.add_argument('--idx_file', type=str, default='wikiart_split.pkl')
    parser.add_argument('--mode', type=str, default='easy', help='choose mode of caption picking (easy/hard)')
    parser.add_argument('--K', type=int, default=3, help='number of nearest neighbor to choose caption from')
    args = parser.parse_args()
    print(args)

    fix_path = lambda p: p if osp.isabs(p) else osp.join(ROOT_DIR, p)
    args.splits_file = fix_path(args.splits_file)
    args.test_splits_file = fix_path(args.test_splits_file)
    args.idx_file = fix_path(args.idx_file)
    args.nn_file = fix_path(args.nn_file)

    with open(args.idx_file, 'rb') as f:
        image_ids = pickle.load(f)
    image_ids = [(unicodedata.normalize('NFD', k), v) for k, v in image_ids]

    with open(args.nn_file, 'rb') as f:
        nearest_neighbors = pickle.load(f)
    nearest_neighbors = nearest_neighbors.astype(int)
    print(f'Nearest neighbor matrix size: {nearest_neighbors.shape}')

    print('generating image to id dict')
    image_id = {}
    id_image = {}
    for item in image_ids:
        image_id[item[0]] = item[1]
        id_image[item[1]] = item[0]
    print('done')
    print('loading splits')
    images_df, train_paints, test_paints = load_image_splits(args.splits_file, args.test_splits_file)
    images_df = images_df.set_index('art_paint')
    print('done')

    test_idxs = np.array(list(map(lambda x: image_id[x], test_paints)))
    train_idxs = np.array(list(map(lambda x: image_id[x], train_paints)))
    print('Calculating Test NN')
    def get_nn(nearest_neighbors, test_idxs, K, train_idxs):
        nn = np.zeros((len(test_idxs), K))
        max_j = 0
        for i, row in enumerate(nearest_neighbors[test_idxs]):
            c = 0
            valid_nn = []
            for j, idx in enumerate(row[1:]):
                if idx in train_idxs:
                    valid_nn.append(idx)
                    c += 1
                    if c >= K:
                        break
            max_j = j if j > max_j else max_j
            if len(valid_nn) < K:
                print(f'WARNING: Instance {i} does not have enough train neighbors')   # If this happens alot just increase number of neighbors in generateNN/gen_neighbor.py
            nn[i, :len(valid_nn)] = valid_nn
        print(f'The biggest neighbor to get test {max_j}')
        return nn
    # test_neighbors = nearest_neighbors[test_idxs, 1:args.K+1]
    test_neighbors = get_nn(nearest_neighbors, test_idxs, args.K, train_idxs)
    print('done')

    def gen_cap(row):
        c_set = []
        for i in row:
            utterances = images_df.loc[id_image[i]]['utterances']  
            c_set.extend(utterances)
        gen_cap = pick_captions(c_set, mode=args.mode)
        return gen_cap
    print('generating captions')
    gen_caps = list(map(gen_cap, test_neighbors))
    gt_caps = list(map(lambda x: [np.random.choice(x)],  images_df.loc[test_paints]['utterances'].values))
    print('done')
    print('scoring captions')
    with open('nnbaseline.pickle', 'wb') as f:
        pickle.dump((gen_caps, gt_caps), f)

    gts = evaluation.PTBTokenizer.tokenize(gt_caps)
    gen = evaluation.PTBTokenizer.tokenize(gen_caps)
    scores, _ = evaluation.compute_scores(gts, gen)

    print(scores)
