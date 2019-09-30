from torch import nn
import torch

import torch.nn.functional as F

from torchtext import data
from torchtext import  datasets
import numpy as np

torch.cuda.current_device()
torch.cuda._initialized=True
LABEL = data.Field()
SENTENCE1=data.Field()
SENTENCE2=data.Field()

fields = {'gold_label':('label',LABEL),'sentence1':('st1',SENTENCE1),'sentence2':('st2',SENTENCE2)}

train_data , test_data = data.TabularDataset.splits(path = 'E:\pycharmproject\FuDanNlpDemo3\source\snli_1.0',train='snli_1.0_train.jsonl',test='snli_1.0_test.jsonl',format='json',fields=fields)

##################应该把sent1和sent2进行pad

neutral_count = 0
contract_count = 0
entail_count = 0
xiahua_count = 0
for i in train_data:

    length1 = len(vars(i)['st1'])
    length2 = len(vars(i)['st2'])
    maxlength = length1 if length1>length2 else length2
    while(maxlength-length1):
        vars(i)['st1'].append('<pad>')
        length1+=1
    while(maxlength-length2):
        vars(i)['st2'].append('<pad>')
        length2+=1
    if vars(i)['label'] == ['neutral']:
        neutral_count +=1
    if vars(i)['label'] == ['contradiction']:
        contract_count+=1
    if vars(i)['label'] == ['entailment'] :
        entail_count +=1
    if vars(i)['label']==['-']:
        xiahua_count+=1


print(f'neutral_count:{neutral_count},contract_count:{contract_count}entailment_count :{entail_count}   xiahua_count:{xiahua_count}')
print(f"合计:  {neutral_count+contract_count+entail_count+xiahua_count}")
print(f"训练集的大小为{len(train_data)}")
for i in test_data:

    length1 = len(vars(i)['st1'])
    length2 = len(vars(i)['st2'])
    maxlength = length1 if length1 > length2 else length2
    while (maxlength - length1):
        vars(i)['st1'].append('<pad>')
        length1 += 1
    while (maxlength - length2):
        vars(i)['st2'].append('<pad>')
        length2 += 1

print(vars(train_data[0]))
print(vars(test_data[0]))
print(f"训练集的大小为{len(train_data)}")
print(f"测试集的大小为{len(test_data)}")

MAX_VOCAB_SIZE = 30000

LABEL.build_vocab(train_data)
SENTENCE1.build_vocab(train_data,max_size=MAX_VOCAB_SIZE, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
SENTENCE2.build_vocab(train_data,max_size=MAX_VOCAB_SIZE, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)

print(SENTENCE1.vocab.itos[0],SENTENCE1.vocab)
print(SENTENCE1.vocab.itos[1],SENTENCE1.vocab[1])
print(f"LABEL词表的大小为：{len(LABEL.vocab)}")

print(f"SENTENCE1词表的大小为:{len(SENTENCE1.vocab)}")
print(f"SENTENCE2词表的大小为:{len(SENTENCE2.vocab)}")

print("LABEL词表包括那些东西：：",LABEL.vocab.stoi)

####建立完词表后  定义batch_size和device(为什么？？因为Iterator里面参数要用到batch_size 和device) 然后建立Iterator
device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')
BATCH_SIZE =128
train_iter , test_iter = data.BucketIterator.splits((train_data,test_data),sort=False,batch_size=BATCH_SIZE,device=device)

for batch in train_iter:
    print (batch)
    break

class ESIM(nn.Module):
    def __init__(self, args):
        super(ESIM, self).__init__()
        self.args = args
        self.dropout = 0.5
        self.hidden_size = args.hidden_size
        self.embeds_dim = args.embeds_dim
        self.pad_idx = args.pad_idx
        self.num_word = args.num_word

        self.embeds = nn.Embedding(self.num_word, self.embeds_dim,padding_idx=self.pad_idx)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size * 8, self.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, args.linear_size),
            nn.ELU(inplace=True),
            # nn.BatchNorm1d(args.linear_size),
            # nn.Dropout(self.dropout),
            # nn.Linear(args.linear_size, args.linear_size),
            # nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size,4)
            # nn.Softmax(dim=-1)
        )

    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))   ###为什么叫mask，就像面具一样，把原来<pad>的值全部用 -inf覆盖了，这样softmax出来就都是0 ，也就是attention中和<pad>相关的注意力都是0
        #mask       batch_size  *seq_len    unsqueeze之后变成batch_size *1 *seq_len

        # weight: batch_size * seq_len * seq_len

        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)   #batch_size *(seq_len)*seq_len 注意矩阵维度不一样的时候的加法
        # '''[2,3,3]+[2,1,3]
        # torch.Size([2, 3, 3])
        # tensor([[[-0.3740, 1.0252, 0.5291],
        #          [-0.2367, 2.1526, -0.2691],
        #          [-1.6116, -1.5094, 0.2614]],
        #
        #         [[-0.8805, 1.8062, 0.2487],
        #          [-0.0205, 0.1859, -0.5070],
        #          [0.2409, 0.5222, 1.0360]]])
        # tensor([[[1.3760, 0.6039, 0.2657]],
        #
        #         [[0.6044, 0.7289, -0.6047]]])
        # tensor([[[1.0020, 1.6291, 0.7947],
        #          [1.1394, 2.7565, -0.0035],
        #          [-0.2356, -0.9055, 0.5270]],
        #
        #         [[-0.2760, 2.5351, -0.3560],
        #          [0.5839, 0.9148, -1.1117],
        #          [0.8453, 1.2511, 0.4312]]])
        # '''
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)    #batch_size*seq_len *seq_len         batch_size * seq_len * hidden_size
        # x_align: batch_size * seq_len * hidden_size
        return x1_align, x2_align


    def submul(self, x1, x2):
        mul = x1 * x2            ###############element-wise 相应元素相乘,论文中是这么写的吗????
        sub = x1 - x2
        return torch.cat([sub, mul], -1)     ####按最后一个维度进行拼接

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, sentence1,sentence2):
        # batch_size * seq_len
        sent1, sent2 = sentence1, sentence2


        mask1, mask2 = sent1.eq(0), sent2.eq(0)    ####这里的mask其实就是原始句子中的<pad>标签把

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)    #batch_size *seq_len * 2*embedding_dim
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)

        return similarity

#使用传统的双向LSTM  好像也可以试试


# class RNN(nn.Module):
#
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
#                  bidirectional, dropout, pad_idx):
#         super().__init__()
#
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
#
#         self.rnn = nn.LSTM(embedding_dim,
#                            hidden_dim,
#                            num_layers=n_layers,
#                            bidirectional=bidirectional,
#                            dropout=dropout)
#
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, text, text_lengths):
#         # text = [sent len, batch size]
#         embedded = self.dropout(self.embedding(text))
#         # embedded = [sent len, batch size, emb dim]
#         # pack sequence
#         packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
#         packed_output, (hidden, cell) = self.rnn(packed_embedded)
#         # unpack sequence
#         output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
#         # output = [sent len, batch size, hid dim * num directions]
#         # output over padding tokens are zero tensors
#         # hidden = [num layers * num directions, batch size, hid dim]
#         # cell = [num layers * num directions, batch size, hid dim]
#         # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
#         # and apply dropout
#         hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
#         # hidden = [batch size, hid dim * num directions]
#         return self.fc(hidden)
#
#



#####################初始化需要哪些东西??? hidden_size   embeds_dim     linear_size

INPUT_DIM1 = len(SENTENCE1.vocab)
INPUT_DIM2 = len(SENTENCE2.vocab)
EMBEDS_DIM = 100
HIDDEN_SIZE = 128
LINEAR_SIZE = 50
OUTPUT_DIM = 3
DROPOUT = 0.5

PAD_IDX =SENTENCE1.vocab.stoi[SENTENCE1.pad_token]

# PAD_IDX2 =SENTENCE2.vocab.stoi[SENTENCE1.pad_token]
#
# print("pad_idx1:",PAD_IDX1)
#
# print("pad_idx2:",PAD_IDX2)

class arg:
    def __init__(self,a,b,c,d,e):
        self.hidden_size = a
        self.embeds_dim = b
        self.linear_size = c
        self.pad_idx = d
        self.num_word =e

NUM_WORD = len(SENTENCE1.vocab)
ARG = arg(HIDDEN_SIZE,EMBEDS_DIM,LINEAR_SIZE,PAD_IDX,NUM_WORD)

model = ESIM(ARG)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

################################准     备      训         练#####################################


pretrained_embedding = SENTENCE1.vocab.vectors
# pretrained_embedding2 = SENTENCE2.vocab.vectors
# print(pretrained_embedding.shape)
# print(pretrained_embedding2.shape)
model.embeds.weight.data.copy_(pretrained_embedding)
UNK_IDX = SENTENCE1.vocab.stoi[SENTENCE1.unk_token]
PAD_IDX = SENTENCE1.vocab.stoi[SENTENCE1.pad_token]

model.embeds.weight.data[UNK_IDX] = torch.zeros(EMBEDS_DIM)
model.embeds.weight.data[PAD_IDX] = torch.zeros(EMBEDS_DIM)

# print(model.embeds.weight.data)

##########################定义优化器   损失函数 ###################################################

import torch.optim as optim
optimizer = optim.Adam(model.parameters())

criterion =nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

#############################################################################
########################################################################


def get_result(preds):  ################输入[batchsize    C ]    返回[batchsize]大小的list
    ######################################用来和Sentiment比较计算准确率
    result = []
    for i in preds:
        i = i.cpu().detach().numpy()
        result.append(int(np.argmax(i)))
    return result

def get_accuracy(preds, y):
    ###多分类这个不知道怎么写！！！！
    """"return accuracy per batch , i.e."""
    result = []
    for i in preds:
        i = i.cpu().detach().numpy()
        result.append(int(np.argmax(i)))
    sum = 0
    for i in range(len(y)):
        if result[i] == y[i]:
            sum = sum + 1
    acc = sum / len(y)
    return acc

# for batch in train_iter:
#     print(batch)


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    count = 0
    for batch in iterator:
        optimizer.zero_grad()
        st1 = batch.st1
        st2 = batch.st2
        st1 = st1.transpose(0, 1)
        st2 = st2.transpose(0, 1)
        predictions = model(st1, st2)
        batch.label = batch.label - 2  # 类别下标只能从0开始  不然各种报错
        # print("Label", batch.label[0])
        # print("prediction",get_result(predictions))
        # print(get_accuracy(predictions,batch.label[0]))
        loss = criterion(predictions, batch.label[0])
        acc = get_accuracy(predictions, batch.label[0])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
        count+=1
        if(count%1000==0):
            print(epoch_acc/count)
            epoch_acc = 0
            count = 0

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in iterator:
            st1, st2 = batch.st1,batch.st2
            st1 = st1.transpose(0, 1)
            st2 = st2.transpose(0, 1)
            predictions = model(st1, st2)
            batch.label = batch.label - 2
            # print(batch.label)
            # print("prediceions:",predictions)
            # print("Label",batch.label[0])
            loss = criterion(predictions, batch.label[0])
            acc = get_accuracy(predictions, batch.label[0])
            epoch_loss += loss.item()
            epoch_acc += acc
            count += 1
            if (count % 1000 == 0):
                print(epoch_acc / count)
                epoch_acc = 0
                count = 0

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


import time


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_iter, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')
    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}| Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')