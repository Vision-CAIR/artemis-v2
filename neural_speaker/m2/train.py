import argparse
import itertools
import multiprocessing
import os
import pickle
import random
from shutil import copyfile

import numpy as np
import torch
from torch.nn import NLLLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import evaluation
from data import (ArtEmis, ArtEmisDetectionsField, DataLoader, EmotionField,
                  RawField, TextField)
from evaluation import Cider, PTBTokenizer
from models.transformer import (MemoryAugmentedEncoder, MeshedDecoder,
                                ScaledDotProductAttentionMemory, Transformer)

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def evaluate_loss(model, dataloader, loss_fn, text_field, emotion_encoder=None):
    # Validation loss
    model.eval()
    if emotion_encoder is not None:
        emotion_encoder.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions, emotions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                if emotion_encoder is not None:
                    emotions = F.one_hot(emotions, num_classes=9)
                    emotions = emotions.type(torch.FloatTensor)
                    emotions = emotions.to(device)
                    enc_emotions = emotion_encoder(emotions)
                    enc_emotions = enc_emotions.unsqueeze(1).repeat(1, detections.shape[1], 1)
                    detections = torch.cat([detections, enc_emotions], dim=-1)
                out = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field, emotion_encoder=None):
    import itertools
    model.eval()
    if emotion_encoder is not None:
        emotion_encoder.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_emos) in enumerate(iter(dataloader)):
            images = images.to(device)
            caps_gt, emotions = caps_emos
            if emotion_encoder is not None:
                emotions = torch.stack([torch.mode(emotion).values for emotion in emotions]) # pick the most frequent emotion
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
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field, emotion_encoder):
    # Training with cross-entropy
    model.train()
    if emotion_encoder is not None:
        emotion_encoder.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions, emotions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)

            if emotion_encoder is not None:
                emotions = F.one_hot(emotions, num_classes=9)
                emotions = emotions.type(torch.FloatTensor)
                emotions = emotions.to(device)
                enc_emotions = emotion_encoder(emotions)
                enc_emotions = enc_emotions.unsqueeze(1).repeat(1, detections.shape[1], 1)
                detections = torch.cat([detections, enc_emotions], dim=-1)

            out = model(detections, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)
    return loss

if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--use_emotion_labels', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    print('Meshed-Memory Transformer Training')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

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
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
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
                                     attention_module_kwargs={'m': args.m}, d_in=2048 + emotion_dim)
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field})
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field})


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
    
    params = list(model.parameters())
    if args.use_emotion_labels:
        params = list(model.parameters()) + list(emotion_encoder.parameters())

    # Initial conditions
    optim = Adam(params, lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    best_cider = .0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                           num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

        train_loss = train_xe(model, dataloader_train, optim, text_field, emotion_encoder)
        writer.add_scalar('data/train_loss', train_loss, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field, emotion_encoder)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field, emotion_encoder)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)
    
        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field, emotion_encoder)
        print("Test scores", scores)
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)
    

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        exit_train = False
        if patience == 5:
            print('patience reached')
            exit_train = True

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
        }, 'saved_models/%s_last.pth' % args.exp_name)
        if args.use_emotion_labels:
            torch.save(emotion_encoder.state_dict(), 'saved_models/%s_emo_last.pth' % args.exp_name)

        if best:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)
            if args.use_emotion_labels:
                copyfile('saved_models/%s_emo_last.pth' % args.exp_name, 'saved_models/%s_emo_best.pth' % args.exp_name)

        if exit_train:
            writer.close()
            break