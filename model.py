import torch 
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import pickle
class EMM(nn.Module):

    def __init__(self,input_size,hidden_size,mem_count=3):
        super(EMM,self).__init__()
        #weight of attention
        self.att_weight = nn.Parameter(torch.Tensor(input_size,input_size))
        self.att_weight.data.uniform_(-0.05,0.05)
        #Attention layer
        self.attention_layer = nn.Sequential(
                                    nn.Linear(7*input_size+2,hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size,1))
                                    #nn.Sigmoid())
        
        #Memory module
        self.mem_layer = nn.GRUCell(input_size,input_size)
        self.mem_count = mem_count

    def dot(self,c,w,q): 
        out = torch.matmul(torch.matmul(c,w),q.permute(0,2,1))
        return out

    def z(self,c,m,q):
        q_input = q.expand(q.shape[0],c.shape[1],q.shape[2])
        m_input = m.expand(m.shape[0],c.shape[1],m.shape[2])
        out = torch.cat((c,m_input,q_input,c*q,c*m,torch.abs(c-q),torch.abs(c-m),self.dot(c,self.att_weight,q),self.dot(c,self.att_weight,m)),2)
        return out
    
    def pad_sequence(self,sequences, batch_first=True):
        s_len = 0
        idx = 0
        for i, sequence in enumerate(sequences):
            if s_len < sequence.size()[0]:
                s_len = sequence.size()[0]
                idx = i
        max_size = sequences[idx].size()
        max_len, trailing_dims = max_size[0], max_size[1:]
        prev_l = max_len
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims

        out_variable = Variable(sequences[idx].data.new(*out_dims).zero_())
        for i, variable in enumerate(sequences):
            length = variable.size(0)
            # use index notation to prevent duplicate references to the variable
            if batch_first:
                out_variable[i, :length, ...] = variable
            else:
                out_variable[:length, i, ...] = variable

        return out_variable 
    
    def forward(self,c,q,len_c):
        m = q
        att_result = []
        for time in range(self.mem_count):
            att_input=self.z(c,m,q)
            batch_size = att_input.shape[0]
            att_input = att_input.view(-1,att_input.shape[2])
            att_scores = self.attention_layer(att_input)
            att_scores = att_scores.view(batch_size,-1,1) 
            att_scores = [nn.functional.softmax(att_score[:len_c[i]],0) for i,att_score in enumerate(att_scores)] 
            att_scores = self.pad_sequence(att_scores)
            att_result.append(att_scores)
            e = (c*att_scores).sum(1)            
            m = self.mem_layer(e,m.squeeze())
            m = m.unsqueeze(1)
        
       
        return m.squeeze(),att_result
      

class DMN(nn.Module):

    def __init__(self,input_size,hidden_size,out_size,ans_count=2,att_hidden_size=100):

        super(DMN, self).__init__()
        #Dropout
        self.dropout = nn.Dropout(0.3)
        #Input Module
        self.input_layer = nn.GRU(input_size,hidden_size,1,batch_first=True)
        #Question Module
        self.question_layer = nn.GRU(input_size,hidden_size,1,batch_first=True)
        #Memory Module
        self.emm = EMM(hidden_size,att_hidden_size)
        #Answer Module
        self.out_layer = nn.Sequential(
                                nn.Linear(hidden_size,out_size),
                                nn.Softmax(1))
        self.ans_layer = nn.GRUCell(hidden_size+out_size,hidden_size)
        self.ans_count = ans_count
   
    def pad_sequence(self,sequences, batch_first=True):
        s_len = 0
        idx = 0
        for i, sequence in enumerate(sequences):
            if s_len < sequence.size()[0]:
                s_len = sequence.size()[0]
                idx = i
        max_size = sequences[idx].size()
        max_len, trailing_dims = max_size[0], max_size[1:]
        prev_l = max_len
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims

        out_variable = Variable(sequences[idx].data.new(*out_dims).zero_())
        for i, variable in enumerate(sequences):
            length = variable.size(0)
            # use index notation to prevent duplicate references to the variable
            if batch_first:
                out_variable[i, :length, ...] = variable
            else:
                out_variable[:length, i, ...] = variable

        return out_variable     
    def forward(self,c,q,c_index,i_state,q_state):
        c = self.pad_sequence(c)
        q = self.pad_sequence(q)
        c = self.dropout(c)
        q = self.dropout(q)
        c_out,c_h = self.input_layer(c,i_state)
        q_out,q_h = self.question_layer(q,q_state)
        q_h = q_h[0,:,:].unsqueeze(1)
        #q_h = q_h.view(q_h.shape[1],q_h.shape[0],q_h.shape[2])
        c_out = [c_out[[i],index,:] for i,index in enumerate(c_index)]
        len_c = [len(c) for c in c_out ]
        c_out = self.pad_sequence(c_out)
        m,att_scores = self.emm(c_out,q_h,len_c)
        for time in range(self.ans_count): 
            y = self.out_layer(m)
            m = self.ans_layer(torch.cat((y,q_h.squeeze()),1),m)

        y = self.out_layer(m)


        return y,att_scores

#from data_utils import Dataset
#import pickle
#from data_utils import to_var
#batch_size=3
#hidden_size = 200
#i_state = Variable(torch.zeros(1,batch_size,hidden_size))
#q_state = Variable(torch.zeros(1,batch_size,hidden_size))

#dataset = pickle.load(open('data/train.pickle','rb'))

#c =[Variable(torch.randn(40,100)),Variable(torch.randn(32,100)),Variable(torch.randn(30,100))]
#q =[Variable(torch.randn(4,100)),Variable(torch.randn(4,100)),Variable(torch.randn(4,100))]
#c_index = [[1,2,3,4,5],[7,8,9,10],[4,5,6]]

#model = DMN(100,200,21)
#y,att_scores = model(c,q,c_index,i_state,q_state)

#print(y)
#print(att_scores)
#for idx, (s, sl, q, a, al) in enumerate(dataset.data_loader('train')):
#    c = [to_var(torch.from_numpy(s_row).type(torch.FloatTensor)) for s_row in s]
#    c = sorted(c,key=len,reverse=True)
#    q = [to_var(torch.from_numpy(q_row).type(torch.FloatTensor)) for q_row in q]
#    c_index = sl
#    y,att_scores = model(c,q,c_index,i_state,q_state)






