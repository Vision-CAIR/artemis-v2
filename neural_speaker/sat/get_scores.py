import torch
import pandas as pd
import itertools
import pickle
import numpy as np
import random
import argparse

from artemis.in_out.basics import unpickle_data
from artemis.utils.vocabulary import Vocabulary
from artemis.evaluation.single_caption_per_image import apply_basic_evaluations

def print_out_some_basic_stats(captions):
    """
    Input: captions dataframe with column names caption
    """
    print('Some basic statistics:')
    mean_length = captions.caption.apply(lambda x: len(x.split())).mean()
    print(f'average length of productions {mean_length:.4}')
    unique_productions = len(captions.caption.unique()) / len(captions)
    print(f'percent of distinct productions {unique_productions:.2}')
    maximizer = captions.caption.mode()
    print(f'Most common production "{maximizer.iloc[0]}"')
    n_max = sum(captions.caption == maximizer.iloc[0]) 
    print(f'Most common production appears {n_max} times -- {n_max/ len(captions):.2} frequency.')
    u_tokens = set()
    captions.caption.apply(lambda x: [u_tokens.add(i) for i in x.split()]);
    print(f'Number of distinct tokens {len(u_tokens)}')
    
    
# needed to make the custom test data same as wikiart 
def split_img_file(row):
    img_file = row['image_file']
    splitted_img_file = img_file.split('/')
    art_style = splitted_img_file[-2]
    painting = splitted_img_file[-1].split('.')[0]
    row['art_style'] = art_style
    row['painting'] = painting
    return row

ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']
EMOTION_TO_IDX = {e: i for i, e in enumerate(ARTEMIS_EMOTIONS)}
IDX_TO_EMOTION = {EMOTION_TO_IDX[e]: e for e in EMOTION_TO_IDX}
POS_NEG_ELSE = {'amusement': 0, 'awe': 0, 'contentment': 0, 'excitement': 0,
                'anger': 1, 'disgust': 1,  'fear': 1, 'sadness': 1,
                'something else': 2}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str, default=None,
                        help='Emotion breakdown filter. one from [dummy, fine, coarse]')
    parser.add_argument('--filter_limit', type=int, default=0,
                        help='threshold where reference captions are removed')
    return parser.parse_args()

def pos_neg_filter_tmp(original_row, target_emo, emo_mapping, threshold):
    row = original_row.copy()
    acceptable_emos = []
    acceptable_caps_pre = []
    acceptable_caps = []
    for i, emo in enumerate(row['emotion']):
        if emo_mapping[IDX_TO_EMOTION[emo]] == emo_mapping[target_emo]:
            acceptable_emos.append(emo)
            acceptable_caps_pre.append(row['references_pre_vocab'][i])
            acceptable_caps.append(row['references'][i])
            
    # Case where there are no acceptable captions 
    if len(acceptable_emos) <= threshold:
        acceptable_emos = acceptable_caps_pre = acceptable_caps = None
    else:
        # if the len is less than 5 duplicate till enough
        if len(acceptable_emos) < 5:
            rem = 5 - len(acceptable_emos)
            duplicating_factor = rem // 2 + rem % 2
            acceptable_emos *= 3 * duplicating_factor
            acceptable_caps_pre *= 3 * duplicating_factor
            acceptable_caps *= 3 * duplicating_factor
        # sample 5 instances from every list
        random.seed(1)
        acceptable_emos, acceptable_caps_pre, acceptable_caps = [list(t) for t in zip(*random.sample(list(zip(acceptable_emos,
                                                                                            acceptable_caps_pre,
                                                                                            acceptable_caps)
                                                                                       ), 5))]
    # assign the results to the row and return it to the dataframe
    row['emotion'] = acceptable_emos
    row['references_pre_vocab'] = acceptable_caps_pre
    row['references'] = acceptable_caps
    return row

def dummy_filter(original_row, target_emo):
    return original_row

def group_gt_annotations(preprocessed_dataframe, vocab):
    df = preprocessed_dataframe
    results = dict()
    for split, g in df.groupby('split'): # group-by split
        g.reset_index(inplace=True, drop=True)
        g = g.groupby(['art_style', 'painting']) # group-by stimulus

        # group utterances / emotions
        # a) before "vocabularization" (i.e., raw)
        refs_pre_vocab_grouped = g['utterance_spelled'].apply(list).reset_index(name='references_pre_vocab')
        # b) post "vocabularization" (e.g., contain <UNK>)
        tokens_grouped = g['tokens_encoded'].apply(list).reset_index(name='tokens_encoded')
        emotion_grouped = g['emotion_label'].apply(list).reset_index(name='emotion')

        assert all(tokens_grouped['painting'] == emotion_grouped['painting'])
        assert all(tokens_grouped['painting'] == refs_pre_vocab_grouped['painting'])

        # decode these tokens back to strings and name them "references"
        tokens_grouped['tokens_encoded'] =\
            tokens_grouped['tokens_encoded'].apply(lambda x: [vocab.decode_print(eval(sent)) for sent in x])
        tokens_grouped = tokens_grouped.rename(columns={'tokens_encoded': 'references'})

        # join results in a new single dataframe
        temp = pd.merge(emotion_grouped, refs_pre_vocab_grouped)
        result = pd.merge(temp, tokens_grouped)
        result.reset_index(drop=True, inplace=True)
        results[split] = result
    return results

args = parse_args()

evaluation_methods = {'bleu', 'cider', 'meteor', 'rouge'}
# top-image dir
wiki_art_img_dir = '/ibex/scratch/mohameys/wiki_art_paintings/rescaled_600px_max_side/'
split = 'test'
gpu_id = 0
device = torch.device("cuda:" + str(gpu_id))
default_lcs_sample = [25000, 800]
# train_sets = ['combined_emo', 'old_large_emo']# , 'combined', 'old_large', 'new', 'old_small']
# train_sets = ['old_small_emo', 'new_emo']
train_sets = ['combined_emo', 'old_large_emo', 'old_small_emo', 'new_emo']
# test_sets = ['all', 'new', 'old', 'ref', '40']
# test_sets = ['all', '40']
test_sets = ['40']

pos_neg_filter = None
if args.filter == 'dummy':
    pos_neg_filter = dummy_filter
elif args.filter == 'fine':
    pos_neg_filter = lambda x, y: pos_neg_filter_tmp(x, y, EMOTION_TO_IDX, args.filter_limit)
elif args.filter == 'coarse':
    pos_neg_filter = lambda x, y: pos_neg_filter_tmp(x, y, POS_NEG_ELSE, args.filter_limit)

print(f'Using granuality: {args.filter}::{pos_neg_filter}')

for exp_name in train_sets:
    for test_set in test_sets:
        # output of preprocess_artemis_data.py
#         references_file = '/home/mohameys/artemis/artemis/artemis/dataset/old_full/train/artemis_gt_references_grouped.pkl'  # evaluating the training sets
        references_file = f'/home/mohameys/artemis/artemis/artemis/dataset/test_{test_set}/test_{test_set}.csv'
        if 'emo' in exp_name:
            vocab_path = '/home/mohameys/artemis/artemis/artemis/dataset/'+'_'.join(exp_name.split('_')[:-1])+'/train/vocabulary.pkl'
        elif 'vanilla' in exp_name:
            vocab_path = '/home/mohameys/artemis/artemis/artemis/dataset/'+'old_full'+'/train/vocabulary.pkl'
        else:
            vocab_path = '/home/mohameys/artemis/artemis/artemis/dataset/'+exp_name+'/train/vocabulary.pkl'
        # the file with the samples
        sampled_captions_file = f'/home/mohameys/artemis/artemis/sampled/SAT_Artemis_{exp_name}_{test_set}.pkl'
        
        txt2emo_clf = None
        txt2emo_vocab = Vocabulary.load(vocab_path)
        
        if 'pkl' in references_file:
            gt_data = next(unpickle_data(references_file))
            train_utters = gt_data['train']['references_pre_vocab']
            gt_data = gt_data[split]     
        else:
            gt_data = pd.read_csv(references_file)
            gt_data = group_gt_annotations(gt_data, txt2emo_vocab)
            gt_data = gt_data[split]  
            train_utters = gt_data['references_pre_vocab']
            
        
        train_utters = list(itertools.chain(*train_utters))  # undo the grouping per artwork to a single large list
        print('Training Utterances', len(train_utters))
        unique_train_utters = set(train_utters)
        print('Unique Training Utterances', len(unique_train_utters))
        print('Images Captioned', len(gt_data))
        
        saved_samples = next(unpickle_data(sampled_captions_file))

        for sampling_config_details, captions, attn in saved_samples:  # you might have sampled under several sampling configurations
            print('Sampling Config:', sampling_config_details)        
            print(f'exp name: {exp_name}_{test_set}')
            print()            
            print_out_some_basic_stats(captions)
            print()
            
            # required to make the custom test data the same as the wikiart test set
            captions = captions.apply(split_img_file, axis=1) if 'csv' in references_file else captions
            
            merged = pd.merge(gt_data, captions)  # this ensures proper order of captions to gt (via accessing merged.captions)
            def compute_metrics(df):
                hypothesis = df.caption
                references = df.references_pre_vocab
                ref_emotions = df.emotion
                metrics_eval = apply_basic_evaluations(hypothesis, references, ref_emotions, txt2emo_clf, txt2emo_vocab, 
                                                       nltk_bleu=False, lcs_sample=default_lcs_sample,
                                                       train_utterances=unique_train_utters,
                                                       methods_to_do=evaluation_methods)
                return pd.DataFrame(metrics_eval)
            
            if pos_neg_filter is not None:
                gdf=merged.groupby(by='grounding_emotion').agg(list)
                emotion_metrics = {}
                for target_emo in gdf.index:
                    s = gdf.loc[target_emo]
                    dft = pd.DataFrame.from_dict(dict(zip(s.index, s.values)))
                    filter_func = lambda x: pos_neg_filter(x, target_emo)
                    dft = dft.apply(filter_func, axis=1)
                    df = dft.dropna()
                    print(f'For Emotion {target_emo}, Samples dropped: {len(dft)-len(df)}')

                    metrics_eval = compute_metrics(df)
                    emotion_metrics[target_emo] = metrics_eval
                # average score per emotion and save the emotion breakdown scores as a .pkl file
                a = pd.Series(np.zeros_like(list(emotion_metrics.values())[0]['mean']))
                for df in emotion_metrics.values():
                    a += df['mean']
                a /= len(emotion_metrics)
                print(a)
                with open(f'emotion_break/{exp_name}_{test_set}_{args.filter}.pkl', 'wb') as f:
                    pickle.dump(emotion_metrics, f)
            else:
                metrics_eval = compute_metrics(merged)
                print(pd.DataFrame(metrics_eval))
                print()
        print('#'*75)