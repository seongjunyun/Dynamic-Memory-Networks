import numpy as np
import nltk
import re
import pickle
import torch
from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
        return Variable(x, volatile=volatile)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.pad = '<pad>'
        #self.unk = '<unk>'
        #self.word2idx[self.pad] = 0
        #self.idx2word[0] = self.pad
        #self.word2idx[self.unk] = 1
        #self.idx2word[1] = self.unk
        self.idx = 0
    def add_words(self,words):
        for word in words:
            if not word in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx+=1        
    def add_word(self,word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx+=1        

class Dataset(object):
    def __init__(self,qa_type,path='data/'):
        self.path = path
        self.qa_type = qa_type
        self.max_len = 0
        self.vocab = Dictionary()
        self.dataset = {}
        self.glove = pickle.load(open('data/glove.pickle','rb'))

    def tokenize(self,line):
        def replace(word):
            return word.replace("''", '"').replace("``", '"')

        def nltk_process(s):
            return list(map(lambda t: replace(t), nltk.word_tokenize(s)))

        def process_unk(word):
            if not word in self.glove:
                return '<unk>'
            else:
                return word

        return list(map(lambda t: process_unk(t),nltk_process(line)))	    

    def preprocess(self,type):
        f = open(self.path+'qa'+str(self.qa_type)+'_'+type+'.txt')
        lines = f.readlines()
        f.close() 
        self.dataset[type] = []

        for line in lines:
            line = line.split('\t')
            #case: story
            if len(line)==1:
                no = line[0].split()[0]
                story = line[0][len(no)+1:]
                no = int(no)
                #start of story
                if no==1:		
                    no_dic = {}
                    stroy_indexes = []
                    index = 0
                    prev_story=''
                
                story = prev_story+' '+story 
                no_dic[no] = index                    
                index+=1
                prev_story = story
            #case: question answer
            else:
                no = line[0].split()[0]
                question = line[0][len(no)+1:]
                answer = line[1]
                ans_loc = list(map(lambda t: no_dic[int(t)],line[2].split()))
                story_tokens = self.tokenize(story.lower())
                story_indexes = [index for index,v in enumerate(story_tokens) if v == '.']
                question = self.tokenize(question.lower())
                answer = self.tokenize(answer.lower())
                self.vocab.add_words(story_tokens)
                self.vocab.add_words(question)
                self.vocab.add_words(answer)
                data = (self.word2idx(story_tokens),story_indexes,self.word2idx(question),self.word2idx(answer),ans_loc)
                self.dataset[type].append(data)
                if len(story_indexes) > self.max_len:
                    self.max_len = len(story_indexes)
                #print(data) 
                #print('story\n {}\n story_indexes: {}, question: {}, answer: {}, ans_loc: {}'.format(story,story_indexes,question,answer,ans_loc)) 
                #print('story\n {}\n story_indexes: {}, question: {}, answer: {}, ans_loc: {}'.format(self.idx2vec(data[0]),story_indexes,self.idx2vec(data[2]),self.idx2vec(data[3]),ans_loc))
                 
                
    def build_vocab(self):
        for (s,s_idx,q,a,a_loc) in self.dataset:

            self.vocab.add_words(s)
            self.vocab.add_words(q)
            self.vocab.add_words(a)

    def word2idx(self,tokens):
        return [self.vocab.word2idx[token] for token in tokens]

    def idx2word(self,idxs):
        return [self.vocab.idx2word[idx] for idx in idxs]

    def idx2vec(self,idxs):
        return np.array([self.glove[word] for word in self.idx2word(idxs)])

    def data_loader(self,mode,batch_size=20):
        
        dataset = self.dataset[mode]        
        batch_s, batch_sl, batch_q, batch_a, batch_al = ([] for _ in range(5))
        self.iters_per_epoch = len(dataset)/batch_size
        
        for data in dataset:
            s, sl, q, a, al = data
            batch_s.append(self.idx2vec(s))
            batch_sl.append(sl)
            batch_q.append(self.idx2vec(q))
            batch_a.append(a)
            batch_al.append(al)
            batch = [(s,sl,q,a,al) for s,sl,q,a,al in zip(batch_s,batch_sl,batch_q,batch_a,batch_al)]
            batch = sorted(batch, key=lambda tup:len(tup[0]),reverse=True)
            s,sl,q,a,al = zip(*batch)
            if len(batch_s) == batch_size:
                yield (s,sl,q,a,al)
                del (batch_s[:], batch_sl[:], batch_q[:],batch_a[:], batch_al[:]) 

        #batch = [(s,sl,q,a,al) for s,sl,q,a,al in zip(batch_s,batch_sl,batch_q,batch_a,batch_al)]
        #batch = sorted(batch, key=lambda tup:len(tup[0]),reverse=True)
        #s,sl,q,a,al = zip(*batch)
        #if len(batch_s) == batch_size:
        #        yield (s,sl,q,a,al)    
        
        

if __name__ == '__main__':

    from data_utils import Dataset

    for i in np.arange(14,20):
        print('start')
        data = Dataset(i+1)
        data.preprocess('train')
        data.preprocess('valid')
        data.preprocess('test')
        pickle.dump(data,open('data/qa'+str(i+1)+'.pickle','wb')) 
        print(i)
#dataset = pickle.load(open('data/qa2.pickle','rb'))

#for idx, (s, sl, q, a, al) in enumerate(dataset.data_loader('train')):
    #print(s[0].shape)
    #print(sl[0])
    #print(q[1])
    #print(dataset.idx2word(a))
