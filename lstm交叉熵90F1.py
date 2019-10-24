import torch
import torch.nn as nn
import torchtext
import torch.optim as optim
import torch.tensor as tensor
from sklearn.metrics import precision_recall_fscore_support as score

import torchtext.data as data
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator
from tqdm import tqdm
from TorchCRF import CRF
from torchtext.datasets.sequence_tagging import  SequenceTaggingDataset
torch.cuda.current_device()
torch.cuda._initialized=True
import time
from sklearn.metrics import f1_score,classification_report
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
###             CONLL 2003的数据格式   word      pos_tag     chunk_tag     ner_tag

filepath = 'E:\pycharmproject\ccccccrf+llllll\source'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORD = data.Field(include_lengths=True)
POS = data.Field()
CHUNK = data.Field()
LABEL = data.Field() #pad_token = 0
##############对于ner任务，提取第四列作为标签就可以了！  其余的都可以不用管   conlleval要求的格式是 word    tag    prediction的格式，只提取word和tag也方便生成txt文件
# fields=[('word', WORD),   ('pos', POS),('chunk',CHUNK ),  ('label', LABEL)]
fields=[('word', WORD),   (None, None),(None,None),  ('label', LABEL)]

train_data,valid_data,test_data = SequenceTaggingDataset.splits(fields=fields,path = filepath,separator=' ',
                                                                train = 'train.txt',
                                                                validation = 'valid.txt',
                                                             test = 'test.txt')
BATCH_SIZE =8
WORD.build_vocab(train_data,vectors='glove.6B.100d',unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
# print(word_frqs[:20])device

train_iter =  data.BucketIterator(dataset = train_data,batch_size=BATCH_SIZE,device=device,repeat=False, sort_key=lambda x:len(x),sort_within_batch=True,shuffle=True)
valid_iter =  data.BucketIterator(dataset = valid_data,batch_size=BATCH_SIZE,device=device,repeat=False, sort_within_batch=True,shuffle=True)
test_iter =  data.Iterator(dataset = test_data,batch_size=BATCH_SIZE,device=device,repeat=False, sort_within_batch=True,shuffle=True)

PAD_IDX = WORD.vocab.stoi[WORD.pad_token]
pad_idx = LABEL.vocab.stoi[LABEL.pad_token]
print(f"LABEL词表的大小为：{len(LABEL.vocab)}")
print(LABEL.vocab.itos[:20])
print(f'{PAD_IDX,pad_idx}')


class BiLstmCrf(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.tag_num = args.tag_num
        self.batch_size = args.batch_size
        self.bidirectional = True
        self.num_layers = args.num_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension,padding_idx= self.pad_index).to(device)

        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim // 2, bidirectional=self.bidirectional,
                            num_layers=self.num_layers).to(device)

        self.hidden2label = nn.Linear(self.hidden_dim, self.tag_num).to(device)



    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(device)

        return h0, c0



    def forward(self, sentence, sent_lengths):
        x = self.embedding(sentence.to(device)).to(device)
        x = pack_padded_sequence(x, sent_lengths.tolist())
        self.hidden = self.init_hidden(batch_size=len(sent_lengths))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)
        assert torch.equal(sent_lengths, new_batch_size.to(device))
        y = self.hidden2label(lstm_out.to(device))
        return y


INPUT_DIM = len(WORD.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
TAG_NUM = 9
N_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = WORD.vocab.stoi[WORD.pad_token]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class arg:
    def __init__(self, a, b, c, d, e,f,g,h):
        self.hidden_dim = a
        self.tag_num = b
        self.batch_size = c
        self.num_layers = d
        self.pad_index = e
        self.dropout = f
        self.vocabulary_size=g
        self.embedding_dim = h

ARG = arg(128,9,64,1,1,0.5,INPUT_DIM,100)
model = BiLstmCrf(ARG)
pretrained_embedding = WORD.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embedding)

UNK_IDX = WORD.vocab.stoi[WORD.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

################################# 开始训练
import torch.optim as optim

optimizer = optim.Adam(model.parameters(),lr=0.02)
# criterion = nn.CrossEntropyLoss(weight = torch.tensor([0.05,1,1,1.5,1.5,1.5,1.5,1.5,1.5]))
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 20

best_valid_loss = float('inf')


def train(model, iterator, optimizer, criterion):
    acc_loss = 0
    epoch_loss = 0
    epoch_f1 = 0

    model.train()

    predresult = tensor([], device=device)
    labelresult = []
    predresult2 = []
    print(model)
    quanju1 = []
    quanju2 = []
    for batch in tqdm(iterator):

        predresult = tensor([], device=device)
        optimizer.zero_grad()
        text, text_length = batch.word
        predictions = model(text, text_length)
        predictions = predictions.reshape(-1, 9)

        label = batch.label - 2
        label = label.view(-1)

        nq = text.ne(1).view(-1)

        label = label.masked_select(nq)

        index = 0
        for i in nq:
            if i == 1:
                predresult = torch.cat([predresult, predictions[index]], dim=0)
                index += 1
        predresult = predresult.view(-1, 9)

        # item_loss = criterion(predresult, label,weight = torch.tensor([1,20,20,20,20,20,20,20,20]))      ####损失打注释的准确率和f1都挺高
        item_loss = criterion(predresult, label)
        acc_loss += item_loss
        item_loss.backward()

        optimizer.step()
        for i in label:
            labelresult.append(i.item())
        for i in predresult:
            predresult2.append(i.argmax(dim=0).item())

        epoch_loss = acc_loss / len(train_iter)
        #
        # micro_score2 = classification_report(labelresult, predresult2, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        quanju1.append(labelresult)
        quanju2.append(predresult2)

    micro_score2 = classification_report(labelresult, predresult2, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    return micro_score2,epoch_loss



def evaluate(model, iterator, optimizer, criterion):
    acc_loss = 0
    epoch_loss = 0

    model.eval()


    labelresult = []
    predresult2 = []
    print(model)
    quanju1 = []
    quanju2 = []
    with torch.no_grad():
        for batch in tqdm(iterator):

            predresult = tensor([], device=device)
            optimizer.zero_grad()
            text, text_length = batch.word

            text = text.transpose(0,1)

            predictions = model(text, text_length)
            ##输入的textshepe为（seqlen,_batch_size）
            print(predictions.shape)
            predictions = predictions.reshape(-1, 9)

            label = batch.label - 2
            label = label.view(-1)

            nq = text.ne(1).view(-1)

            label = label.masked_select(nq)

            index = 0
            for i in nq:
                if i == 1:
                    predresult = torch.cat([predresult, predictions[index]], dim=0)
                    index += 1
            predresult = predresult.view(-1, 9)

            # item_loss = criterion(predresult, label,weight = torch.tensor([1,20,20,20,20,20,20,20,20]))
            item_loss = criterion(predresult, label)

            acc_loss += item_loss
            item_loss.backward()

            optimizer.step()
            for i in label:
                labelresult.append(i.item())
            for i in predresult:
                predresult2.append(i.argmax(dim=0).item())

            epoch_loss = acc_loss / len(train_iter)
            #
            # micro_score2 = classification_report(labelresult, predresult2, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            quanju1.append(labelresult)
            quanju2.append(predresult2)

        micro_score2 = classification_report(labelresult, predresult2, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        return micro_score2,epoch_loss





for epoch in range(N_EPOCHS):

    start_time = time.time()
    acc_loss = 0
    epoch_loss = 0
    epoch_f1 = 0

    model.train()

    predresult = tensor([],device=device)
    labelresult = []
    predresult2=[]
    print(model)
    quanju1=[]
    quanju2=[]
    for batch in tqdm(valid_iter):

        predresult = tensor([], device=device)
        optimizer.zero_grad()
        text, text_length = batch.word
        predictions = model(text, text_length)
        predictions=predictions.reshape(-1,9)

        label = batch.label-2
        label = label.view(-1)

        nq = text.ne(1).view(-1)

        label = label.masked_select(nq)

        index = 0
        for i in  nq:
            if i ==1:
                predresult =torch.cat([predresult,predictions[index]],dim=0)
                index +=1
        predresult = predresult.view(-1,9)

        item_loss =criterion(predresult,label)
        acc_loss += item_loss
        item_loss.backward()

        optimizer.step()
        for i in label:
            labelresult.append(i.item())
        for i in predresult:
            predresult2.append(i.argmax(dim=0).item())

        epoch_loss = acc_loss/len(valid_iter)
        #
        # micro_score2 = classification_report(labelresult, predresult2, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        quanju1.append(labelresult)
        quanju2.append(predresult2)


    micro_score2 = classification_report(labelresult, predresult2, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    print(micro_score2)





