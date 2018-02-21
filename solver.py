import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle
from torch.autograd import grad
from torch.autograd import Variable
from data_utils import Dataset
from model import DMN

class Solver(object):

    def __init__(self, config):
        #Dataset
        self.dataset = pickle.load(open('data/qa'+config.qa_type+'.pickle','rb'))
        self.qa_type = config.qa_type 
        # Model hyper-parameters
        self.hidden_size = config.hidden_size
        self.out_size = len(self.dataset.vocab.word2idx)
        self.input_size = config.input_size

        # Hyper-parameters
        self.lr = config.lr
        
        # Training settings
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.alpha = 0
        self.early_stop = 7
        self.stop_count = 0 
        
        # Validation settings
        self.best_ppl = 100000
        self.best_ans = 0
        self.al_acc = 0
        self.ans_acc = 0

        # Test settings
        self.test_model = config.test_model

        # Path
        self.model_save_path = config.model_save_path
        
        # Step size
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step

        # Build model
        self.build_model()

    def build_model(self):
        # Define DMN
        self.model = DMN(self.input_size,self.hidden_size,self.out_size)
        
        if torch.cuda.is_available():
            self.model.cuda()

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Print networks
        self.print_network(self.model, 'DMN')
    
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self):
        
        #Start training
        for epoch in range(self.num_epochs):
            self.model.train()
            for i, (s, sl, q, a, al) in enumerate(self.dataset.data_loader('train',self.batch_size)):
                
                # Load batch data
                ans = a
                c = [self.to_var(torch.from_numpy(s_row).type(torch.FloatTensor)) for s_row in s]
                q = [self.to_var(torch.from_numpy(q_row).type(torch.FloatTensor)) for q_row in q]
                c_index = sl
                a = self.to_var(torch.from_numpy(np.array(a)).type(torch.LongTensor))
                try:
                    al = self.to_var(torch.from_numpy(np.array(al)).type(torch.LongTensor))
                except:
                    print(al)
                # Compute loss with answer location and answer 
                self.model.zero_grad()
                i_state = self.to_var(torch.zeros(1,self.batch_size,self.hidden_size))
                q_state = self.to_var(torch.zeros(1,self.batch_size,self.hidden_size))
                y,att_scores = self.model(c,q,c_index,i_state,q_state)                
                if self.al_acc > 0.99 or epoch > 20:
                    self.alpha = 1
                for j in range(al.size()[1]):
                    if j==0:
                        loss = F.cross_entropy(att_scores[j].squeeze(),al[:,j].contiguous().view(-1))
                    else:
                        loss += F.cross_entropy(att_scores[j].squeeze(),al[:,j].contiguous().view(-1))
                loss += self.alpha*F.cross_entropy(y,a.view(-1))

                # Backward and Optimize
                loss.backward()
                self.optimizer.step()

                # Compute accuracy
                att_idx = []
                al_count = 0
                for j in range(al.size()[1]):
                    _,idx = torch.max(att_scores[j],1)
                    idx = idx.squeeze()
                    att_idx.append(idx)
                    al_count += (idx==al[:,j]).sum().data[0]
                _,y_idx = torch.max(y,1)
                ans_count = 0
                for j in range(len(ans)):
                    if ans[j][0] == y_idx[j].data[0]:
                        ans_count += 1

                # Logging
                acc = {}
                acc['al'] = al_count/(self.batch_size*al.size()[1])
                acc['ans'] = ans_count/self.batch_size

                # Print out log info
                if (i+1) % self.log_step == 0:
                    print ('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f, acc_al: %.2f, ans_al: %.2f' %
                        (epoch+1, self.num_epochs, i+1, self.dataset.iters_per_epoch,
                        loss.data[0], np.exp(loss.data[0]),acc['al']*100,acc['ans']*100))
                
                # Save model checkpoints
                #if (i+1) % self.model_save_step == 0:
                #    torch.save(self.model.state_dict(),
                #        os.path.join(self.model_save_path, '{}_{}_{}_model.pth'.format(self.qa_type,epoch+1, i+1)))
            # Validate
            self.validate(epoch)

            if self.stop_count > self.early_stop or self.best_ans > 0.99:
                break


            
    def validate(self,epoch):
        #Validate
        self.model.eval()
        al_count = 0
        ans_count = 0
        val_loss = 0
        for i, (s, sl, q, a, al) in enumerate(self.dataset.data_loader('valid',self.batch_size)):
            
            # Load batch data
            ans = a
            c = [self.to_var(torch.from_numpy(s_row).type(torch.FloatTensor)) for s_row in s]
            q = [self.to_var(torch.from_numpy(q_row).type(torch.FloatTensor)) for q_row in q]
            c_index = sl
            a = self.to_var(torch.from_numpy(np.array(a)).type(torch.LongTensor))
            al = self.to_var(torch.from_numpy(np.array(al)).type(torch.LongTensor))

            # Compute loss with answer location and answer 
            self.model.zero_grad()
            i_state = self.to_var(torch.zeros(1,self.batch_size,self.hidden_size))
            q_state = self.to_var(torch.zeros(1,self.batch_size,self.hidden_size))
            y,att_scores = self.model(c,q,c_index,i_state,q_state)                
            if self.al_acc > 0.99 or epoch > 20:
                    self.alpha = 1                
            for j in range(al.size()[1]):
                if j==0:
                    loss = F.cross_entropy(att_scores[j].squeeze(),al[:,j].contiguous().view(-1))
                else:
                    loss += F.cross_entropy(att_scores[j].squeeze(),al[:,j].contiguous().view(-1))
            loss += self.alpha*F.cross_entropy(y,a.view(-1))
            val_loss += loss.data[0]

            # Compute accuracy
            att_idx = []
            for j in range(al.size()[1]):
                _,idx = torch.max(att_scores[j],1)
                idx = idx.squeeze()
                att_idx.append(idx)
                al_count += (idx==al[:,j]).sum().data[0]
            _,y_idx = torch.max(y,1)
            for j in range(len(ans)):
                if ans[j][0] == y_idx[j].data[0]:
                    ans_count += 1

        # Logging
        acc = {}
        acc['al'] = al_count/(self.batch_size*self.dataset.iters_per_epoch*al.size()[1])
        acc['ans'] = ans_count/(self.batch_size*self.dataset.iters_per_epoch)

        self.al_acc = acc['al']
        self.ans_acc = acc['ans']
        # Print out log info
        print ('Validation:  Loss: %.3f, Perplexity: %5.2f, acc_al: %.2f, ans_al: %.2f' %
               (val_loss/self.dataset.iters_per_epoch,np.exp(val_loss/self.dataset.iters_per_epoch),acc['al']*100,acc['ans']*100))
        
        # Save model checkpoints
        if val_loss < self.best_ppl:
            self.best_ppl = val_loss
            torch.save(self.model.state_dict(), 'best_model_{}.pth'.format(self.qa_type))
        
        if acc['ans'] >= self.best_ans:
            self.best_ans = acc['ans']
            self.stop_count = 0
        else:
            if self.alpha > 0:
                self.stop_count += 1
    def test(self)
        #Test
        self.model.eval()
        al_count = 0
        ans_count = 0
        val_loss = 0
        for i, (s, sl, q, a, al) in enumerate(self.dataset.data_loader('test',self.batch_size)):
            
            # Load batch data
            ans = a
            c = [self.to_var(torch.from_numpy(s_row).type(torch.FloatTensor)) for s_row in s]
            q = [self.to_var(torch.from_numpy(q_row).type(torch.FloatTensor)) for q_row in q]
            c_index = sl
            a = self.to_var(torch.from_numpy(np.array(a)).type(torch.LongTensor))
            al = self.to_var(torch.from_numpy(np.array(al)).type(torch.LongTensor))

            # Compute loss with answer location and answer 
            self.model.zero_grad()
            i_state = self.to_var(torch.zeros(1,self.batch_size,self.hidden_size))
            q_state = self.to_var(torch.zeros(1,self.batch_size,self.hidden_size))
            y,att_scores = self.model(c,q,c_index,i_state,q_state)                
            if self.al_acc > 0.99 or epoch > 20:
                    self.alpha = 1                
            for j in range(al.size()[1]):
                if j==0:
                    loss = F.cross_entropy(att_scores[j].squeeze(),al[:,j].contiguous().view(-1))
                else:
                    loss += F.cross_entropy(att_scores[j].squeeze(),al[:,j].contiguous().view(-1))
            loss += self.alpha*F.cross_entropy(y,a.view(-1))
            val_loss += loss.data[0]

            # Compute accuracy
            att_idx = []
            for j in range(al.size()[1]):
                _,idx = torch.max(att_scores[j],1)
                idx = idx.squeeze()
                att_idx.append(idx)
                al_count += (idx==al[:,j]).sum().data[0]
            _,y_idx = torch.max(y,1)
            for j in range(len(ans)):
                if ans[j][0] == y_idx[j].data[0]:
                    ans_count += 1

        # Logging
        acc = {}
        acc['al'] = al_count/(self.batch_size*self.dataset.iters_per_epoch*al.size()[1])
        acc['ans'] = ans_count/(self.batch_size*self.dataset.iters_per_epoch)

        # Print out log info
        print ('Test:  Loss: %.3f, Perplexity: %5.2f, acc_al: %.2f, ans_al: %.2f' %
               (val_loss/self.dataset.iters_per_epoch,np.exp(val_loss/self.dataset.iters_per_epoch),acc['al']*100,acc['ans']*100))

