import torch
from torch.autograd import Variable
import torch.nn as nn
from torchtext import data
from torchtext import datasets
import pandas as pd
import torch.nn.functional as F

SEED = 1234
import torch.optim as optim
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator

torch.manual_seed(SEED)
import random
import tqdm
import numpy as np

torch.cuda.current_device()
torch.cuda._initialized = True
# device = torch.device("cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT = data.Field(tokenize='spacy', include_lengths=True)
# LABEL = data.LabelField(dtype = torch.int)
LABEL = Field(sequential=False)
fields = [("PhraseId", None), ('SentenceId', None), ("Phrase", TEXT), ("Sentiment", LABEL)]
#####skip_header 就是跳过第一行标题(每个数据的名字)的意思
# 这里train_data test_data还是正常的数据，但是BucketIterator是什么呢

fields_for_test = [("PhraseId", None), ('SentenceId', None), ("Phrase", TEXT)]

train_data = TabularDataset(path='E:/pycharmproject/FuDanNlpDemo1/source/train.tsv',
                            format='tsv', skip_header=True, fields=fields)
test_data = TabularDataset(path='E:/pycharmproject/FuDanNlpDemo1/source/test.tsv',
                           format='tsv', skip_header=True, fields=fields_for_test)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))
# print((train_data[0]).Phrase)   #测试train_data中的数据是怎样的

MAX_VOCAB_SIZE = 25000
TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
# print(len(train_data),len(valid_data),len(test_data))
# print(len(LABEL.vocab))
print((LABEL.vocab.itos))

train_iter, valid_iter = BucketIterator.splits((train_data, valid_data), batch_sizes=(256, 256),
                                               sort_key=lambda x: len(x.Phrase), sort_within_batch=True, repeat=False,
                                               shuffle=True,
                                               device=device)
test_iter = Iterator(test_data, batch_size=16, device=device, repeat=False, sort_key=lambda x: len(x.Phrase),
                     sort_within_batch=True)

###########################################注意      一般划分测试集和验证集用BucketIterator
##############################################################测试集一般用  Iterator

BATCH_SIZE = 256
INPUT_DIM = len(TEXT.vocab)  # 这样输入是不是相当于one-hot   输入也太大了         可能准确率低是因为用了one hot 向量


###########################################假如用 embedding 的话效果应该会好很多

# for batch in train_iter:
#     x,y = batch.Phrase
#     print(x)
#     print('+++++++++++++++++++++++++++++++++++')
#     print(y)
#####################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@自己写的垃圾类，先注释了再说
# class RNN(nn.Module):
#     def __init__(self,vocab_size, embedding_dim, hidden_dim, output_dim,n_layers,bidirectional,dropout,pad_idx):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim,padding_idx=pad_idx)
#         self.rnn = nn.LSTM(embedding_dim, hidden_dim,num_layers=n_layers,bidirectional=bidirectional,dropout=dropout)
#         self.fc = nn.Linear(2*hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, text,text_lengths):
#
#         """Lengths (Tensor) – list of sequences lengths of each batch element."""
#         #####这里的输入一定要是
#         # [sent  length , batch  size]   之前已经把batch.Phrase中的向量打印出来检查过,确实是
#         # [sent length,batch size ]
#         # 在输入的时候应该先把维度打印出来进行检查
#         embeded = self.dropout(self.embedding(text))
#
#         #pack sequence
#         packed_embeded = nn.utils.rnn.pack_padded_sequence(embeded,text_lengths)
#
#         packed_output,(hidden,cell) = self.rnn(packed_embeded)
#
#         # embeded = self.embedding(text)
#         # embedded = [sent len, batch size, emb dim]
#         output, output_lengths =  nn.utils.rnn.pad_packed_sequence(packed_output)
#         # output = [sent len, batch size, hid dim]
#         # hidden = [1, batch size, hid dim]
#         hidden =self.dropout(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1))
#
#         return self.fc(hidden)  # 也就是输出是   (batch_size ，output_dim)这么一个向量 而在这里
#     # output_dim设置为 5
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        # self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2,30),
            nn.Linear(30,output_dim)
        )
        # self.fc = nn.Sequential(
        #     nn.BatchNorm1d(self.hidden_size * 8),
        #     nn.Linear(self.hidden_size * 8, args.linear_size),
        #     nn.ELU(inplace=True),
        #     nn.BatchNorm1d(args.linear_size),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(args.linear_size, args.linear_size),
        #     nn.ELU(inplace=True),
        #     nn.BatchNorm1d(args.linear_size),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(args.linear_size, 4)
        #     # nn.Softmax(dim=-1)
        # )

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent len, batch size, emb dim]
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 5
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


pretrained_embedding = TEXT.vocab.vectors
print(pretrained_embedding.shape)

model.embedding.weight.data.copy_(pretrained_embedding)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)

import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

model = model.to(device)

criterion = criterion.to(device)


###################################################################################
###################################################################################

def get_accuracy(preds, y):
    ###多分类这个不知道怎么写！！！！
    """"return accuracy per batch , i.e."""
    result = []
    for i in preds:
        # i = i.cpu().detach().numpy()
        # result.append(int(np.argmax(i)))
        result.append(int(torch.argmax(i)))
    sum = 0
    for i in range(len(y)):
        if result[i] == y[i]:
            sum = sum + 1
    acc = sum / len(y)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_length = batch.Phrase
        predictions = model(text, text_length)
        batch.Sentiment = batch.Sentiment - 1  # 类别下标只能从0开始  不然各种报错     数据中的Sentiment是从1开始的
        loss = criterion(predictions, batch.Sentiment)
        acc = get_accuracy(predictions, batch.Sentiment)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.Phrase
            predictions = model(text, text_lengths)
            batch.Sentiment = batch.Sentiment - 1
            loss = criterion(predictions, batch.Sentiment)
            acc = get_accuracy(predictions, batch.Sentiment)
            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


import time


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 20

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')
    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}| Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
#     ####################################################5分类问题,准确率只有   51%#################################         !!!!!!!!!
#
#     ########################没救了，不管怎么改都只有51的准确率！！！！！！！！！！！！是不是输出的问题呢
#
# #


model.load_state_dict(torch.load('tut2-model.pt'))

# test_loss, test_acc = evaluate(model, test_iter, criterion)

# print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

import spacy

nlp = spacy.load('en')


def get_result(preds):  ################输入[batchsize    C ]    返回[batchsize]大小的list
    ######################################用来和Sentiment比较计算准确率
    result = []
    for i in preds:
        i = i.cpu().detach().numpy()
        result.append(int(np.argmax(i)))
    return result


def predict_sentiment(model, sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = model(tensor, length_tensor)
    prediction = get_result(prediction)
    return prediction


print("of")
print(predict_sentiment(model, "of"))

print("what")
print(predict_sentiment(model, "what"))

print("the film is good")
print(predict_sentiment(model, "the file is good"))

print("the film is very terrible")
print(predict_sentiment(model, 'the file is very terrible'))
