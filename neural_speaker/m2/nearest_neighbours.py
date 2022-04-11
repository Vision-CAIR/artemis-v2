import random
import evaluation
import torch
from tqdm import tqdm
import pickle
import numpy as np
from skimage import io
from skimage.transform import resize
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import os

from evaluation.bleu import Bleu

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, img_names):
        'Initialization'
        self.img_names = img_names

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_names)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.img_names[index]
        img = read_img(ID)
        x = np.moveaxis(img, [0, 1, 2], [-2, -1, -3])
        x = torch.from_numpy(x).float()
        # Load data and get label
        return x
        
def read_img(name):
    path = '/home/haydark/wiki_art_paintings/rescaled_600px_max_side/'
    try:
        img = io.imread(path + name +'.jpg')
        return resize(img, (224, 224))
    except:
        style =  name.split('/')[0] + '/'
        painting = name.split('/')[-1] +'.jpg'
        img_name = painting.split('_')[-1]
        dir_content = os.listdir(path+style)
        for dname in dir_content:
            if img_name == dname.split('_')[-1]:
                img = io.imread(path+style+dname)
                return resize(img, (224, 224))

def load_dataset():
    path = '/home/haydark/meshed-memory-transformer/annotations/'
    splits = pd.read_csv(path + 'panos_2020_split_with_spelled_utterances.csv')
    splits['art_paint'] = splits[['art_style', 'painting']].agg('/'.join, axis=1)
    images_df = splits.groupby('art_paint')['utterance'].apply(list).reset_index(name='utterances')
    train_paints = splits[splits['split']=='train']['art_paint'].unique()
    test_paints = splits[splits['split']=='test']['art_paint'].unique()
    
    return images_df, train_paints, test_paints

def load_embeddings(model, split):

    
    embeds = []
    
    with tqdm(desc='NN Baseline', unit='it', total=len(split)) as pbar:
        with torch.no_grad():
            for x in split:
                x = x.to(device)
                out = vgg16(x)
                out = out.cpu()
                embeds.append(out.squeeze())
                pbar.update()
            
    return torch.cat(embeds, dim=0)


if __name__ == '__main__':
   
    image_df, train_paints, test_paints = load_dataset()

    # Parameters
    params = {'batch_size': 64,
            'shuffle': False,
            'num_workers': 4}


    # Generators
    training_set = Dataset(train_paints)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    test_set = Dataset(test_paints)
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    vgg16 = models.vgg16(pretrained=True)
    new_classifier = nn.Sequential(*list(vgg16.classifier.children())[:-3])
    vgg16.classifier = new_classifier
    metric = Bleu()
    vgg16 = vgg16.to(device)

    print("Extracting support imgs ...")
    support_set = load_embeddings(vgg16, training_generator)
    
    print("Extracting support imgs ...")
    query_set = load_embeddings(vgg16, test_generator)
    
    nncaps = {'gts': {}, 'gen' : {}}
    with tqdm(desc='NN Baseline', unit='it', total=len(test_paints)) as pbar:
        for img_id, query in enumerate(query_set):
            
            dist = torch.norm(support_set - query.unsqueeze(0), dim=1, p=None)
            knn = dist.topk(3, largest=False)

            c_set = []
            for i in knn.indices:
                utterances = image_df[image_df['art_paint'] == train_paints[i.item()]]['utterances'].values[0]
                c_set.extend(utterances)
                
            #lex_sim_scores = []
            '''
            for i,c in enumerate(c_set):
                caps = {'gen':{}, 'gts':{}}
                caps['gen'][i] = [c]
                caps['gts'][i] = [c_start for c_start in c_set if c != c_start]
                gts = evaluation.PTBTokenizer.tokenize(caps['gts'])
                gen = evaluation.PTBTokenizer.tokenize(caps['gen'])

                score, _ = metric.compute_score(gts, gen)
                lex_sim_scores.append(np.mean(score))
            '''
            #gen_cap = [c_set[np.argmax(lex_sim_scores)]]
            gen_cap = [c_set[np.random.randint(0, len(c_set))]]
            utterances = image_df[image_df['art_paint'] == test_paints[img_id]]['utterances'].values[0]
            
            print('Generated: ', gen_cap)
            print('Ground Truth ', utterances)

            nncaps['gen'][test_paints[img_id]] = gen_cap
            nncaps['gts'][test_paints[img_id]] = utterances

            pbar.update()

    with open('nnbaseline.pickle', 'wb') as f:
        pickle.dump(nncaps,f)

    gts = evaluation.PTBTokenizer.tokenize(nncaps['gts'])
    gen = evaluation.PTBTokenizer.tokenize(nncaps['gen'])
    scores, _ = evaluation.compute_scores(gts, gen)

    print(scores)