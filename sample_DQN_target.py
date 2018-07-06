# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(user)s
"""

# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(user)s
"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
from Connect_four_environment import *
#np.random.seed(41)
#torch.manual_seed(47)
import math
from collections import namedtuple
import copy 


 
class Agent(nn.Module):
    
    def __init__(self,epsilon=.05,gamma=.9,learning_rate=.001,replay_mem_size=1000):
        super(Agent, self).__init__()
        self.epsilon=epsilon
        self.gamma=gamma
        self.replay_mem_size=replay_mem_size
        self.fc1=nn.Linear(42,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,7)
        self.bn_1=nn.BatchNorm1d(128)
        self.bn_2=nn.BatchNorm1d(128)
        
        
        
        #pdb.set_trace()
        '''
        self.layer1=nn.Sequential(nn.Linear(42,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,7),
            nn.Softmax())
        '''
        '''
        self.layer1=nn.Sequential(nn.Conv2d(1, 16, kernel_size=(4,3), stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16))
            #nn.Linear(16*8,7),
            #nn.Softmax())
        
        self.layer2=nn.Sequential(nn.Linear(16*8,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Linear(128,7),nn.Softmax())
        
        '''

        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(4,3), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU())
        
        self.conv_fc=nn.Linear(2,7)
        
        
        self.opt=optim.RMSprop(self.parameters(),lr=learning_rate)
        
        self.run_type=0
        
    
    
    def forward(self,x):
        
        if self.run_type=='train':
            self.train()
        if self.run_type==0:
            sys.exit("ERRORRRRRR")
        else:
            self.eval()
        
        
        #x=F.relu(self.fc1(x))
        #x=F.relu(self.fc2(x))
        #x=x.reshape(-1,42)
        
        ##x=x.reshape(-1,1,6,7)
        #print(x.shape)
        ##x=self.layer1(x)
        
        ##x=x.reshape(x.size(0),-1)
        
        ##x=self.layer2(x)
        
        x=self.network_forward(x)
        
        #x=x.reshape(-1,42)
        '''
        x=F.relu(self.fc1(x))
        x=self.bn_1(x.view(-1,128))
        x=F.relu(self.fc2(x))
        x=self.bn_2(x)
        
        x=F.sigmoid(self.fc3(x))
        '''
        return x
    
    def epsilon_greedy(self,x):
        if np.random.rand()<self.epsilon:
            x=torch.tensor(np.random.rand(7))
        else:
            
            if self.run_type=='train':
                self.train()
            else:
                self.eval()
            
            
            
            
            x=self.forward(x)
        
        return x
    
    
    def optimize(self,epochs,batch_size,target_Q_network):
        
        self.opt.zero_grad()
        self.agent_data=Agent_Dataset(self,target_Q_network)
        self.agent_data.format_Q_data()
        
        n_of_batches=math.ceil(len(self.agent_data.agent.memory)/batch_size)
        data_loader=DataLoader(dataset=self.agent_data,batch_size=batch_size,shuffle=True)
        
        self.run_type='train'
        
        
        for i in range(epochs):
            for i_batch,batch in enumerate(data_loader):
                self.opt.zero_grad()
                
                outputs=self.forward(batch[0])
                targets=batch[1]
                
                loss = F.smooth_l1_loss(outputs,targets)
                
                
                loss.backward()
                self.opt.step()
            #print(f'epoch {1+i}/{epochs}')
            print(f'loss: {float(loss)}')
        


class Agent_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, agent,agent_target):
        'Initialization'
        self.agent=agent
        self.agent_target=agent_target
        self.len=len(agent.memory)
    def format_Q_data(self):

        self.agent.run_type='eval'
        chosen_actions=self.agent.memory[:,1]
        rewards=np.array(self.agent.memory[:,3],dtype=np.float32)
        
        
        boards_unrolled=np.hstack(self.agent.memory[:,0]).T
        boards_unrolled=torch.tensor(boards_unrolled,dtype=torch.float32)
        
        boards_unrolled_next=np.hstack(self.agent.memory[:,2]).T
        boards_unrolled_next=torch.tensor(boards_unrolled_next,dtype=torch.float32)
        
        Q_values=self.agent.forward(boards_unrolled) # batch_size x 7
        target_Q_values=Q_values.clone()
        Q_value_next_max=self.agent_target.forward(boards_unrolled_next).max(dim=1)[0]
        
        Q_value_next_max=Variable(Q_value_next_max.data,requires_grad=False)
        
        
        target_Q_value=rewards+self.agent.gamma*Q_value_next_max
        target_Q_values[np.arange(len(chosen_actions)),chosen_actions.astype(np.int)]= target_Q_value
        target_Q_values=Variable(target_Q_values.data,requires_grad=False)
        
        self.X=boards_unrolled
        self.target_Q_values=target_Q_values

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.X[index],self.target_Q_values[index]



class conv_agent(Agent):
    
    
    def __init__(self,epsilon=.05,gamma=.9,learning_rate=.001,replay_mem_size=1000):
        super(conv_agent, self).__init__()
        
        self.epsilon=epsilon
        self.gamma=gamma
        self.replay_mem_size=replay_mem_size
        self.opt=optim.RMSprop(self.parameters(),lr=learning_rate)
        
        self.layer1=nn.Sequential(nn.Conv2d(1, 16, kernel_size=(4,3), stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            )
            #nn.Linear(16*8,7),
            #nn.Softmax())
        
        self.layer2=nn.Sequential(nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16)
            )
        
        
        self.layer3=nn.Sequential(nn.Linear(16*8,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Linear(128,7),nn.Softmax())
        
    def network_forward(self,arr):
        
        #pdb.set_trace()
        arr=arr.reshape(-1,1,6,7)
        
        arr=self.layer1(arr)
        
        arr=self.layer2(arr)
        
        arr=arr.reshape(arr.size(0),-1)
        
        arr=self.layer3(arr)
        
        return arr
    
    
    
class vanilla_agent(Agent):
    
    def __init__(self,epsilon=.05,gamma=.9,learning_rate=.001,replay_mem_size=1000):
        super(vanilla_agent, self).__init__()
        
        self.epsilon=epsilon
        self.gamma=gamma
        self.replay_mem_size=replay_mem_size
        self.opt=optim.RMSprop(self.parameters(),lr=learning_rate)
        
        
        self.layer1=nn.Sequential(nn.Linear(42,128),nn.ReLU(),nn.BatchNorm1d(128))
        self.layer2=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.BatchNorm1d(128))
        self.layer3=nn.Sequential(nn.Linear(128,7),nn.Softmax())
        
        
        
    def network_forward(self,arr):
        
        arr=arr.reshape(-1,42)
        
        arr=self.layer1(arr)
        
        #arr=arr.reshape(arr.size(0),-1)
        
        arr=self.layer2(arr)
        
        arr=self.layer3(arr)
        
        return arr






if __name__ == "__main__":
    
    '''
    agent1=Agent();agent2=Agent()
    g=Game(agent1,agent2)
    g.play_game(n_of_turns=300)
    '''
    
    replay_mem=2000
    
    
    epsilon_start=.9
    epsilon_end=.3
    
    epsilon_step=(epsilon_end-epsilon_start)/replay_mem
    
    
    learning_rate_start=.01
    learning_rate_end=.001
    
    learning_rate_step=(learning_rate_end-learning_rate_start)/replay_mem
    
    
    
    
    
    targ_update=3
    '''
    agent1=Agent(epsilon=epsilon_start,gamma=.95,learning_rate=learning_rate_start);
    agent2=Agent(epsilon=epsilon_start,gamma=.95,learning_rate=learning_rate_start)
    g=Game(agent1,agent2)
    
    agent1.epsilon=epsilon_start
    agent2.epsilon=epsilon_start
    
    '''
    
    
    
    agent1=conv_agent(epsilon=epsilon_start,gamma=.96,learning_rate=learning_rate_start);
    agent2=conv_agent(epsilon=epsilon_start,gamma=.96,learning_rate=learning_rate_start)
    agent1_target=copy.deepcopy(agent1);agent1_target.run_type='eval'
    agent2_target=copy.deepcopy(agent2);agent2_target.run_type='eval'
    
    
    g=Game(agent1,agent2)
    
    agent1.epsilon=epsilon_start
    agent2.epsilon=epsilon_start
    
    
    
    
    
    
    
    
    
    for i in range(replay_mem):
        #print(agent1.gamma)
        g.play_game(n_of_turns=2000)
# =============================================================================
#         agent1_data=Agent_Dataset(agent=agent1,agent_target=agent1_target)
#         agent2_data=Agent_Dataset(agent=agent2,agent_target=agent2_target)
#         agent1_data.format_Q_data()
#         agent2_data.format_Q_data()
# =============================================================================
        
        if replay_mem % 1==0:
            agent1.optimize(2,400,agent1_target)
        else:
            agent2.optimize(2,400,agent2_target)
        
        if replay_mem % targ_update==0:
            agent1_target=copy.deepcopy(agent1);agent1_target.run_type='eval'
            agent2_target=copy.deepcopy(agent2);agent2_target.run_type='eval'
# =============================================================================
#         print('agent1 lr:' + str(agent1.opt.param_groups[0]['lr']))
#         print('agent2 lr:' + str(agent2.opt.param_groups[0]['lr']))
#         
#         print('agent1 epsilon:' + str(agent1.epsilon))
#         print('agent2 epsilon:' + str(agent2.epsilon))
# =============================================================================
        
        print(f'replay_mem {1+i}/{replay_mem}............epsilon: {agent1.epsilon}......learning_rate: {agent1.opt.param_groups[0]["lr"]}....gamma:{agent1.gamma}')
        
        agent1.opt.param_groups[0]['lr']+=learning_rate_step
        agent2.opt.param_groups[0]['lr']+=learning_rate_step
        
        agent1.epsilon+=epsilon_step
        agent2.epsilon+=epsilon_step

    


#SmoothL1Loss

#how to train when chosen action is random, and not on-policy
#how to deal with 
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
'''

    def create_mini_batch(self,memory,mb_size=32):
        
    
        
        mb_indices=np.random.choice((self.replay_mem_size+1),mb_size,replace=False)
        
        mini_batch=memory[mb_indices,:]
        
        
        return mini_batch
        
    
        def train_mini_batch(self,memory,mb_size,gamma):
        
        #criterion=nn.SmoothL1Loss()
        
        
        
        mini_batch=self.create_mini_batch(memory,mb_size)
        mb_size=len(mini_batch)
        
            
        #loss=criterion(Q_values,target_Q_values)
        loss = F.smooth_l1_loss(Q_values,target_Q_values)
        loss.backward()
        opt.step()
        #print(loss)
'''





# =============================================================================
# if network_type=='convolution':
#             
#             self.layer1=nn.Sequential(nn.Conv2d(1, 16, kernel_size=(4,3), stride=1, padding=0),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(16),
#                 )
#                 #nn.Linear(16*8,7),
#                 #nn.Softmax())
#             
#             self.layer2=nn.Sequential(nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=0),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(16)
#                 )
#             
#             
#             self.layer3=nn.Sequential(nn.Linear(16*8,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Linear(128,7),nn.Softmax())
#             
#             def reshape(self,arr):
#                 return arr.reshape(-1,1,6,7)
#         
#         if network_type=='vanilla':
#         
#             self.layer1=nn.Sequential(nn.Linear(42,128),nn.ReLU(),nn.BatchNorm1d(128))
#             self.layer2=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.BatchNorm1d(128))
#             self.layer3=nn.Sequential(nn.Linear(128,7),nn.Softmax())
#         
# 
#             def reshape(self,arr):
#                 return arr.reshape(arr.size(0),-1)
# =============================================================================
