import argparse
import pickle
import random

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

import evaluation
from data import (ArtEmis, ArtEmisDetectionsField, DataLoader, EmotionField,
                  RawField, TextField)
from models.transformer import (MemoryAugmentedEncoder, MeshedDecoder,
                                ScaledDotProductAttentionMemory, Transformer)

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def predict_captions(model, dataloader, text_field, emotion_encoder=None):
    import itertools
    if emotion_encoder is not None:
        emotion_encoder.eval()
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        
        for it, (images, caps_emo_pair) in enumerate(iter(dataloader)):
            images = images.to(device)
            caps_gt, emotions = caps_emo_pair
            if emotion_encoder is not None:
                emotions = torch.stack([torch.mode(emotion).values for emotion in emotions])
                emotions = F.one_hot(emotions, num_classes=9)
                emotions = emotions.type(torch.FloatTensor)
                emotions = emotions.to(device)
                enc_emotions = emotion_encoder(emotions)
                enc_emotions = enc_emotions.unsqueeze(1).repeat(1, images.shape[1], 1)
                images = torch.cat([images, enc_emotions], dim=-1)

            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    store_dict = {'gen': gen,'gts': gts} 
    with open('test_results.pickle', 'wb') as f:
        pickle.dump(store_dict, f)

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')

    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--use_emotion_labels', type=bool, default=False)

    args = parser.parse_args()

    print('Meshed-Memory Transformer Evaluation')

    # Pipeline for image regions
    image_field = ArtEmisDetectionsField(detections_path=args.features_path, max_detections=50)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Pipeline for emotion
    emotions = [
        'amusement', 'awe', 'contentment', 'excitement', 
        'anger', 'disgust', 'fear', 'sadness', 'something else'
        ]
    emotion_field = EmotionField(emotions=emotions)

    # Create the dataset
    dataset = ArtEmis(image_field, text_field, emotion_field, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    # Model and dataloaders
    emotion_dim = 0
    emotion_encoder = None
    if args.use_emotion_labels:
        emotion_dim = 10
        emotion_encoder = torch.nn.Sequential(
            torch.nn.Linear(9, emotion_dim)
            )
        emotion_encoder.to(device)
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40}, d_in=2048 + emotion_dim)
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    fname = 'saved_models/%s_best.pth' % args.exp_name
    data = torch.load(fname)
    model.load_state_dict(data['state_dict'])

    if emotion_encoder is not None:
        emotion_encoder.to(device)
        fname = 'saved_models/%s_emo_best.pth' % args.exp_name
        data = torch.load(fname)
        emotion_encoder.load_state_dict(data)

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, text_field, emotion_encoder)
    print(scores)