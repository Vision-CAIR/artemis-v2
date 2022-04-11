import torch
import pandas as pd 
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import time
import datetime
import random
import argparse

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True, help="Output file name")
parser.add_argument("--model", required=True, default='bert', help="pretrained model name")
parser.add_argument("--model_size", required=True, default='base', help="pretrained model size")
parser.add_argument("--dataset", required=True, help="dataset to predict the emotions")
parser.add_argument("--dataset_name", required=True, help="dataset name to save the modified dataset")

args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

EMOTION_ID = {}
ID_EMOTION = {}
with open('/ibex/scratch/mohameys/text_to_emotions/data/emotions.txt', 'r') as f:
    i = 0
    for line in f:
        emotion = line.split('\n')[0]
        EMOTION_ID[emotion] = i
        ID_EMOTION[i] = emotion
        i += 1
num_classes = len(np.unique(list(EMOTION_ID.values())))
assert num_classes == i, f'wrong number of classes in the emotion to id dict. i:{i}, num_classes:{num_classes}'
print(f'Number of unique emotion categories: {num_classes}')

df_train = pd.read_csv(args.dataset)
sentences_train = df_train['utterance']

print('Dataset Loaded ......')

if args.model == 'bert':
    BERT_version = 'bert-'+args.model_size+'-cased'
    Pretrained_tokenizer = BertTokenizer
    Pretrained_model = BertForSequenceClassification
elif args.model == 'roberta':
    BERT_version = 'roberta-'+args.model_size
    Pretrained_tokenizer = RobertaTokenizer
    Pretrained_model = RobertaForSequenceClassification
else:
    raise ValueError(f'model {args.model} is not implemented')

# BERT_version = 'bert-large-cased'
print('Start Tokenizing ......')
tokenizer = Pretrained_tokenizer.from_pretrained(BERT_version, padding_side='right')

MAX_LEN = 128

tokenized_data_train = tokenizer(sentences_train.to_list(), add_special_tokens=True, max_length=MAX_LEN,
                            truncation=True, padding='max_length', return_tensors='pt', return_attention_mask=True)

print('Finished Tokenizing ......')

train_inputs, train_masks = tokenized_data_train['input_ids'], tokenized_data_train['attention_mask']


batch_size = 32
print('Building Dataloaders ......')
train_dataset = TensorDataset(train_inputs, train_masks)
train_sampler = SequentialSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
print('Done ......')

print('Loading Model ......')
model = Pretrained_model.from_pretrained(
    args.model_dir,
    num_labels = num_classes, 
    output_attentions = False,
    output_hidden_states = False,
)
model.to(device)
print('Done ......')

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

print("")
t0 = time.time()
model.eval()
predicted_labels = np.zeros((len(train_dataset), num_classes))

for step, batch in enumerate(train_dataloader):

    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    with torch.no_grad():
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)
    
    logits = outputs[0]
 
    logits = logits.detach().cpu().numpy()

    predicted_labels[step*batch_size : (step+1)*batch_size] = logits

    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
print("  evaluation took: {:}".format(format_time(time.time() - t0)))
        
df_train['go_emotions'] = pd.Series(predicted_labels.tolist())
dataset_name = args.dataset_name
df_train.to_csv(f'/ibex/scratch/mohameys/text_to_emotions/artemis_analysis/data/{dataset_name}_go_emotions.csv', index=False)
