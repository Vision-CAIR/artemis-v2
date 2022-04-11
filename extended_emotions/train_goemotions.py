import torch
import torch.nn.functional as F
import pandas as pd 
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, RobertaTokenizer, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random

def flat_accuracy(preds, labels):
    preds = (1 / (1 + np.exp(-preds)))
    pred_flat = np.round(preds).astype(int)
    labels_flat = labels
    true_labels = np.where(labels_flat > 0)
    pos_recall = np.sum((pred_flat == labels_flat)[true_labels]) / len(true_labels[0])
    weighted_recall = np.sum(pred_flat == labels_flat) / (labels_flat.shape[0] * labels_flat.shape[1])
    return weighted_recall, pos_recall

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out_file", required=True, help="Output file name")
parser.add_argument("-m", "--model", required=True, default='bert', help="pretrained model name")
parser.add_argument("-s", "--model_size", required=True, default='base', help="pretrained model size")
args = parser.parse_args()
out_file = args.out_file

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
EMOTION_ID = {}
ID_EMOTION = {}
with open('data/emotions.txt', 'r') as f:
    i = 0
    for line in f:
        emotion = line.split('\n')[0]
        EMOTION_ID[emotion] = i
        ID_EMOTION[i] = emotion
        i += 1

num_classes = len(np.unique(list(EMOTION_ID.values())))
assert num_classes == i, f'wrong number of classes in the emotion to id dict. i:{i}, num_classes:{num_classes}'
print(f'Number of unique emotion categories: {num_classes}')

df_train = pd.read_csv('data/train.tsv', sep='\t', header=None, names=['utterance', 'emotion', 'id'])
df_train['emotion'] = df_train['emotion'].apply(lambda x: x.split(',')).apply(lambda x: [int(i) for i in x])
sentences_train = df_train['utterance']
labels_train = df_train['emotion'].values
labels_pt_train = torch.zeros((labels_train.shape[0], num_classes))
for i, emo_list in enumerate(labels_train):
    for emo in emo_list:
        labels_pt_train[i, emo] = 1

df_val = pd.read_csv('data/dev.tsv', sep='\t', header=None, names=['utterance', 'emotion', 'id'])
df_val['emotion'] = df_val['emotion'].apply(lambda x: x.split(',')).apply(lambda x: [int(i) for i in x])
sentences_val = df_val['utterance']
labels_val = df_val['emotion'].values
labels_pt_val = torch.zeros((labels_val.shape[0], num_classes))
for i, emo_list in enumerate(labels_val):
    for emo in emo_list:
        labels_pt_val[i, emo] = 1

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

tokenized_data_val = tokenizer(sentences_val.to_list(), add_special_tokens = True, max_length = MAX_LEN, 
                        truncation=True, padding='max_length', return_tensors='pt', return_attention_mask=True)

print('Finished Tokenizing ......')

train_inputs, train_masks, train_labels = tokenized_data_train['input_ids'], tokenized_data_train['attention_mask'], labels_pt_train
validation_inputs, validation_masks, validation_labels = tokenized_data_val['input_ids'], tokenized_data_val['attention_mask'], labels_pt_val

batch_size = 32
print('Building Dataloaders ......')
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

validation_dataset = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_dataset)
validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=batch_size)
print('Done ......')

print('Loading Model ......')
model = Pretrained_model.from_pretrained(
    BERT_version,
    num_labels = num_classes, 
    output_attentions = False,
    output_hidden_states = False,
)
model.to(device)
print('Done ......')

lr = 2e-5
optimizer = AdamW(model.parameters(),
                  lr = lr,
                  eps = 1e-8 
                )

print(f'Adam learning rate: {lr}')

epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

loss_values = []
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    total_loss = []
    total_acc = []
    pos_recall = []
    model.train()
    for step, batch in enumerate(train_dataloader):

      
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
      
        optimizer.zero_grad()  
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)

        logits = outputs[0]
        loss = F.binary_cross_entropy_with_logits(logits, b_labels)

        total_loss.append(loss.item())
 
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        train_acc = flat_accuracy(logits, label_ids)
        total_acc.append(train_acc[0])
        pos_recall.append(train_acc[1])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.     Train_Loss: {:.5f} Train_acc: {:.3f} Emotion_recall: {:.3f}'.format(step, len(train_dataloader), elapsed, np.mean(total_loss), np.mean(total_acc), np.mean(pos_recall)))

    
    avg_train_loss = np.mean(total_loss)  
    avg_train_acc = np.mean(total_acc)    
    avg_emo_recall = np.mean(pos_recall)            

    loss_values.append(avg_train_loss)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Average training accuracy: {0:.2f}".format(avg_train_acc))
    print("  Average training emotion recall: {0:.2f}".format(avg_emo_recall))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    print("")
    print("Running Validation...")
    t0 = time.time()
    model.eval()
    
    eval_loss, eval_accuracy, eval_recall = 0, 0, 0
    nb_eval_steps = 0
    for batch in validation_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2]
        
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.numpy()
        
        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy[0]
        eval_recall += tmp_eval_accuracy[1]
        # Track the number of batches
        nb_eval_steps += 1
    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Emotion recall: {0:.2f}".format(eval_recall/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
print("")
print("Training complete!")
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

model.save_pretrained(f'/ibex/scratch/mohameys/text_to_emotions/go_models/{out_file}_trained_bert_{current_time}')