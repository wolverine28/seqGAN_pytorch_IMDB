# -*- coding:utf-8 -*-

import os
import random
import math

import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from word_correction import tokenizer

from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from torchtext.legacy import data, datasets
import torchtext
import dill
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 200
GENERATED_NUM = 10000
# PRE_EPOCH_NUM = 120
PRE_EPOCH_NUM = 60
SEQ_LEN = 50
opt.cuda = 0
IR = 50

EVAL_FILE = 'eval.data'

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Genrator Parameters
g_emb_dim = 100
g_hidden_dim = 40
# g_sequence_len = 100

# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

d_dropout = 0.75
d_num_class = 2



def generate_samples(model, batch_size, generated_num):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, SEQ_LEN).cpu().data.numpy().tolist()
        samples.extend(sample)
    return np.array(samples)

def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for data in tqdm(data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data.text[1:])
        if opt.cuda:
            data = data.cuda()
        pred = model.forward(data)
        loss = criterion(pred[:-1], data.view(-1)[1:])
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return math.exp(total_loss / total_words)

def discriminator_train_epoch(model, real_loader, fake_loader, criterion, optimizer):
    total_loss = 0.
    total_words = 0.

    fake_iter = iter(fake_loader)
    for i, real_data in enumerate(tqdm(real_loader, mininterval=2, desc=' - Training', leave=False)):
        try:
            fake_data = next(fake_iter)
        except:
            fake_iter = iter(fake_loader)
            fake_data = next(fake_iter)

        fake_data = Variable(fake_data[0])
        real_data = Variable(real_data.text[:,1:-1])

        data = torch.vstack((fake_data,real_data))
        target = [0 for _ in range(len(fake_data))] +\
                        [1 for _ in range(len(real_data))]
        target = Variable(torch.Tensor(target)).type(torch.long)

        if opt.cuda:
                data, target = data.cuda(), target.cuda()

        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / total_words


def eval_epoch(model, data_iter, criterion):
    total_loss = 0.
    total_words = 0.
    with torch.no_grad():
        for data in tqdm(data_iter, mininterval=2, desc=' - Training', leave=False):
            data = Variable(data.text)
            if opt.cuda:
                data = data.cuda()
            pred = model.forward(data)
            loss = criterion(pred[:-1], data.view(-1)[1:])
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)

    assert total_words > 0  # Otherwise NullpointerException
    return math.exp(total_loss / total_words)


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.bool)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)
        return loss


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    ###############################################################################
    glove = torchtext.vocab.GloVe(name='6B',dim=100)
    print(len(glove.itos)) #400000
    print(glove.vectors.shape)


    TEXT = data.Field(sequential=True, batch_first=True, lower=True,init_token='<sos>', eos_token='<eos>',
                    fix_length=SEQ_LEN+2, tokenize=tokenizer)
    LABEL = data.Field(sequential=False, batch_first=True)

    trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(trainset,max_size=30000,vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(trainset)

    # with open("log/TEXT.Field","wb")as f:
    #      dill.dump(TEXT,f)

    # with open("log/LABEL.Field","wb")as f:
    #      dill.dump(LABEL,f)

    # torch.save(trainset.examples,'./log/trset_orig')
    # torch.save(testset.examples,'./log/tsset_orig')

    # trainset_examples = torch.load('./log/trset_orig')
    # testset_examples = torch.load('./log/tsset_orig')

    # with open("log/TEXT.Field","rb")as f:
    #     TEXT=dill.load(f)
    # with open("log/LABEL.Field","rb")as f:
    #     LABEL=dill.load(f)

    # trainset = data.Dataset(trainset_examples,{'text':TEXT,'label':LABEL})
    # testset = data.Dataset(testset_examples,{'text':TEXT,'label':LABEL})

    print('훈련 샘플의 개수 : {}'.format(len(trainset)))
    print('테스트 샘플의 개수 : {}'.format(len(testset)))
    # print(vars(trainset[0]))
    positive_subset = [i for i in trainset if vars(i)['label']=='pos']
    negative_subset = [i for i in trainset if vars(i)['label']=='neg']

    # positive_count = int(len(negative_subset)/IR)
    positive_count = 2000
    positive_subset = np.random.choice(positive_subset,positive_count).tolist()
    trainset = positive_subset


    trainset = data.Dataset(trainset,{'text':TEXT,'label':LABEL})
    

    VOCAB_SIZE = len(TEXT.vocab)
    n_classes = 2
    print('단어 집합의 크기 : {}'.format(VOCAB_SIZE))
    print('클래스의 개수 : {}'.format(n_classes))
    # print(TEXT.vocab.stoi)


    train_loader = data.Iterator(dataset=trainset, batch_size = BATCH_SIZE)
    ###############################################################################
    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # pretrained_embeddings = TEXT.vocab.vectors
    # print(pretrained_embeddings.shape)
    # generator.emb.weight.data.copy_(pretrained_embeddings)

    # unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
    # pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    # init_idx = TEXT.vocab.stoi[TEXT.init_token]
    # eos_idx = TEXT.vocab.stoi[TEXT.eos_token]

    # generator.emb.weight.data[unk_idx] = torch.zeros(g_emb_dim)
    # generator.emb.weight.data[pad_idx] = torch.zeros(g_emb_dim)
    # generator.emb.weight.data[init_idx] = torch.zeros(g_emb_dim)
    # generator.emb.weight.data[eos_idx] = torch.zeros(g_emb_dim)

    # print(generator.emb.weight.data)


    print(' '.join([TEXT.vocab.itos[i] for i in generator.sample(1, SEQ_LEN)[0]]))
    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters(),lr=0.01)
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    print('Pretrain with MLE ...')
    for epoch in range(PRE_EPOCH_NUM):
        loss = train_epoch(generator, train_loader, gen_criterion, gen_optimizer)
        print('Epoch [%d] Model Loss: %f'% (epoch, loss))

        loss = eval_epoch(generator, train_loader, gen_criterion)
        print('Epoch [%d] True Loss: %f' % (epoch, loss))

        print(' '.join([TEXT.vocab.itos[i] for i in generator.sample(1, SEQ_LEN)[0]]))
         

    # Pretrain Discriminator
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters(),lr=1e-4)
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    print('Pretrain Discriminator ...')
    for epoch in range(5):
        fake_samples = generate_samples(generator, BATCH_SIZE, GENERATED_NUM)
        
        fake_samples = torch.Tensor(fake_samples).long()
        fake_dataset = TensorDataset(fake_samples)
        fake_loader = DataLoader(fake_dataset,batch_size=BATCH_SIZE)

        for _ in range(3):
            loss = discriminator_train_epoch(discriminator, train_loader, fake_loader, dis_criterion, dis_optimizer)
            print('Epoch [%d], loss: %f' % (epoch, loss))

    # Adversarial Training
    rollout = Rollout(generator, 1)
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters(),lr=1e-4)
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    gen_criterion = nn.NLLLoss(reduction='sum')
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters(),lr=1e-4)
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        for it in range(1):
            samples = generator.sample(BATCH_SIZE, SEQ_LEN)
            zeros = Variable(torch.tensor(2).repeat((BATCH_SIZE, 1)).type(torch.LongTensor))
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view(-1)

            # calculate the reward
            input_rewards = Variable(torch.cat([zeros, samples.data], dim = 1).contiguous())
            rewards = rollout.get_reward(input_rewards, 1, discriminator)
            rewards = Variable(torch.Tensor(rewards))
            rewards = torch.exp(rewards).contiguous().view((-1,))
            if opt.cuda:
                rewards = rewards.cuda()
            prob = generator.forward(inputs)
            loss = gen_gan_loss(prob, targets, rewards)
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            loss = eval_epoch(generator, train_loader, gen_criterion)
            print('Epoch [%d] True Loss: %f' % (total_batch, loss))
            print(' '.join([TEXT.vocab.itos[i] for i in generator.sample(1, SEQ_LEN)[0]]))

        rollout.update_params()

        for _ in range(3):
            fake_samples = generate_samples(generator, BATCH_SIZE, GENERATED_NUM)
            
            fake_samples = torch.Tensor(fake_samples).long()
            fake_dataset = TensorDataset(fake_samples)
            fake_loader = DataLoader(fake_dataset,batch_size=BATCH_SIZE)

            for _ in range(1):
                loss = discriminator_train_epoch(discriminator, train_loader, fake_loader, dis_criterion, dis_optimizer)
                print('Epoch [%d], Discriminator loss: %f' % (total_batch, loss))
                
if __name__ == '__main__':
    main()
