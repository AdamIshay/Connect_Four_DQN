# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(user)s
"""

import numpy as np
import torch
import torch.nn as nn
import itertools
from itertools import islice

from sample_DQN import *
import pdb

class Game():
    
    def __init__(self,agent1,agent2):
        self.columns=7
        self.rows=6
        self.agent1=agent1
        self.agent2=agent2
        self.p1=1
        self.p2=-1
        self.players={'p1':-1,'p2':1}
        self.new_game()
        self.vert,self.horz,self.diag1,self.diag2=self.create_filters()
        self.p1_memory=[[],[],[],[]]
        self.p2_memory=[[],[],[],[]]
        self.p1_move_count=0
        self.p2_move_count=0
    
    def new_game(self):
        self.column_counter=np.zeros((1,self.columns),dtype='int')
        self.board=np.zeros((self.rows,self.columns),dtype='int')
        self.board_tensor=torch.tensor(self.board,dtype=torch.float32)
        self.start=0 if np.random.random() > .5 else 1
        self.p1_history={'state':[],'action':[],'next_state':[],'reward':[]}
        self.p2_history={'state':[],'action':[],'next_state':[],'reward':[]}
        self.turn=islice(itertools.cycle(self.players),self.start,None)
        self.current_turn=next(self.turn)
        
    def create_filters(self): 
        
        #vertical filter
        vert=nn.Conv2d(1,1,kernel_size=(4,1),padding=0,bias=False)
        vert.weight=torch.nn.Parameter(torch.tensor(([1,1,1,1]),dtype=torch.float32).view(1,1,4,1))
        #horizontal filter
        horz=nn.Conv2d(1,1,kernel_size=(1,4),padding=0,bias=False)
        horz.weight=torch.nn.Parameter(torch.tensor(([1,1,1,1]),dtype=torch.float32).view(1,1,1,4))
        #diagonal filter #1
        diag1=nn.Conv2d(1,1,kernel_size=(4,4),padding=0,bias=False)
        diag1_array=torch.zeros((4,4))
        
        for i in range(len(diag1_array)):
            diag1_array[i,len(diag1_array)-1-i]=1
        diag1.weight=torch.nn.Parameter(diag1_array.view(1,1,4,4))
        
        #diagonal filter #2
        diag2=nn.Conv2d(1,1,kernel_size=(4,4),padding=0,bias=False)
        diag2_array=torch.zeros((4,4))
        
        for i in range(len(diag2_array)):
            diag2_array[i,i]=1
        diag2.weight=torch.nn.Parameter(diag2_array.view(1,1,4,4))
        
        return vert,horz,diag1,diag2
    
    def choose_move(self):
        
        if self.current_turn=='p1':
            choices=self.agent1.epsilon_greedy(self.board_tensor.view(42).clone())
        else:
            choices=self.agent2.epsilon_greedy(self.board_tensor.view(42).clone())
        
        
        choices_np=choices.data.numpy()
        min_value=choices_np.min()-1
        invalid_indexes=np.where(((self.column_counter<6)*1)==0)[1]
        invalid_value=min_value.copy()
        #print(f'{choices_np} ,\n {invalid_indexes}\n {self.column_counter}')
        
        if choices_np.size> 7:
            pass;#pdb.set_trace()
            
        #pdb.set_trace()
        choices_np=choices_np.reshape(1,7)
        choices_np[0,invalid_indexes]=invalid_value
        
        #choices=torch.tensor(choices_np)
        
        return np.argmax(choices_np,axis=1)
    
    def first_move(self):
        
        pick=self.choose_move(); board_piece=self.players[self.current_turn]
        #print(self.current_turn,pick)
        
        if self.current_turn=='p1':
            self.p1_history['state'].append(self.board.copy().astype(np.float32))
            self.p1_history['action'].append(np.float32(pick))
            self.p1_history['reward'].append(np.float32(0))
            self.p1_move_count+=1;#print('p1_move_count is ' +str(self.p1_move_count));
        else:
            self.p2_history['state'].append(self.board.copy().astype(np.float32))
            self.p2_history['action'].append(np.float32(pick))
            self.p2_history['reward'].append(np.float32(0))
            self.p2_move_count+=1;#print('p2_move_count is ' +str(self.p2_move_count))
        self.board[5-self.column_counter[0,pick],pick]=board_piece
        self.board_tensor[5-self.column_counter[0,pick],pick]=board_piece
        self.column_counter[0,pick]+=1
        #print(self.board);
        self.current_turn=next(self.turn)
        
    def move(self):
        " Starts on the current player's turn"
        
        
        pick=self.choose_move(); board_piece=self.players[self.current_turn]
        #print(self.current_turn,pick)
        if self.current_turn=='p1':
            self.p1_history['state'].append(self.board.copy().astype(np.float32))
            self.p1_history['action'].append(np.float32(pick))
            self.p1_move_count+=1;#print('p1_move_count is ' +str(self.p1_move_count))
        else:
            self. p2_history['state'].append(self.board.copy().astype(np.float32))
            self.p2_history['action'].append(np.float32(pick))
            self.p2_move_count+=1;#print('p2_move_count is ' +str(self.p2_move_count))
            
            
        
        assert self.column_counter[0,pick]<6,'column full'
        self.board[5-self.column_counter[0,pick],pick]=board_piece
        self.board_tensor[5-self.column_counter[0,pick],pick]=board_piece
        self.column_counter[0,pick]+=1
        #print(self.board);
        if self.current_turn=='p1':
            self.p2_history['next_state'].append(self.board.copy().astype(np.float32))
            self.p1_history['reward'].append(np.float32(0))
        else:
            self.p1_history['next_state'].append(self.board.copy().astype(np.float32))
            self.p2_history['reward'].append(np.float32(0))
        
      
        self.current_turn=next(self.turn)
        
        #takes in a column to put piece in
        
    def show_board(self):
        s_board=self.board.copy()
        temp_board=np.where(s_board>0,'X',s_board);
        temp_board=np.where(s_board<0,'O',temp_board);
        temp_board=np.where(s_board==0,'.',temp_board);
        print(temp_board)
        
    def get_bool_board(self):
        return 1*(self.board<0),1*(self.board>0)
    
    def check_wins(self):
        """
        Takes in the board and checks if there are any wins by applying 
        4 convolutions for each way you can get four in a row (vertical,
        horizontal, diagonal with positive slope, and diagonal with 
        negative slope.
        """
        p1_board,p2_board=self.get_bool_board()
        p1_board=torch.tensor(p1_board,dtype=torch.float32).view(1,1,6,7)
        p2_board=torch.tensor(p2_board,dtype=torch.float32).view(1,1,6,7)
        
        
        p1_check=np.zeros((4,1))
        p1_check[0]=(self.vert(p1_board)>=4).sum()
        p1_check[1]=(self.horz(p1_board)>=4).sum()
        p1_check[2]=(self.diag1(p1_board)>=4).sum()
        p1_check[3]=(self.diag2(p1_board)>=4).sum()
        
        if p1_check.sum()>=1:
            #print('player one wins')
            return 'p1'
            
            
        p2_check=np.zeros((4,1))
        p2_check[0]=(self.vert(p2_board)>=4).sum()
        p2_check[1]=(self.horz(p2_board)>=4).sum()
        p2_check[2]=(self.diag1(p2_board)>=4).sum()
        p2_check[3]=(self.diag2(p2_board)>=4).sum()
        
        if p2_check.sum()>=1:
            #print('player two wins')
            return 'p2'
        
        return None
    
    def first_add_to_memory(self):
        
        
        assert len(self.p1_history['state'])==len(self.p1_history['action'])==len(self.p1_history['next_state'])==len(self.p1_history['reward'])
        assert len(self.p2_history['state'])==len(self.p2_history['action'])==len(self.p2_history['next_state'])==len(self.p2_history['reward'])
        
        self.p1_memory[0]=[i.reshape(42,1) for i in self.p1_history['state']]
        self.p1_memory[1]=self.p1_history['action']
        self.p1_memory[2]=[i.reshape(42,1) for i in self.p1_history['next_state']]
        self.p1_memory[3]=self.p1_history['reward']
        
        self.p2_memory[0]=[i.reshape(42,1) for i in self.p2_history['state']]
        self.p2_memory[1]=self.p2_history['action']
        self.p2_memory[2]=[i.reshape(42,1) for i in self.p2_history['next_state']]
        self.p2_memory[3]=self.p2_history['reward']
        
        
    def add_to_memory(self):
        
        assert len(self.p1_history['state'])==len(self.p1_history['action'])==len(self.p1_history['next_state'])==len(self.p1_history['reward'])
        assert len(self.p2_history['state'])==len(self.p2_history['action'])==len(self.p2_history['next_state'])==len(self.p2_history['reward'])
        
        
        self.p1_memory[0]=self.p1_memory[0]+[i.reshape(42,1) for i in self.p1_history['state']]
        self.p1_memory[1]=self.p1_memory[1]+self.p1_history['action']
        self.p1_memory[2]=self.p1_memory[2]+[i.reshape(42,1) for i in self.p1_history['next_state']]
        self.p1_memory[3]=self.p1_memory[3]+self.p1_history['reward']
        
        
        self.p2_memory[0]=self.p2_memory[0]+[i.reshape(42,1) for i in self.p2_history['state']]
        self.p2_memory[1]=self.p2_memory[1]+self.p2_history['action']
        self.p2_memory[2]=self.p2_memory[2]+[i.reshape(42,1) for i in self.p2_history['next_state']]
        self.p2_memory[3]=self.p2_memory[3]+self.p2_history['reward']
        
        
    def play_game(self,n_of_turns=300):
        self.agent1.run_type='eval'
        self.agent2.run_type='eval'
        self.new_game()
        self.p1_move_count=0
        self.p2_move_count=0
        self.p1_memory=[[],[],[],[]]
        self.p2_memory=[[],[],[],[]]
        self.first_move()
        
        
        for i in range(41):
            self.move()
            
            if i >4:
                if self.check_wins():
                    if self.current_turn=='p1':
                        
                        self.p2_history['reward'][-1]=np.float32(1)
                        self.p1_history['reward'][-1]=np.float32(-1)
                        self.p2_history['next_state'].append(self.board.copy().astype(np.float32))
                        
                    else:
                        self.p1_history['reward'][-1]=np.float32(1)
                        self.p2_history['reward'][-1]=np.float32(-1)
                        self.p1_history['next_state'].append(self.board.copy().astype(np.float32))
                    
                    self.first_add_to_memory()
                    
                    break;
                       
        while True:
            if self.p1_move_count>=n_of_turns and self.p2_move_count>=n_of_turns:
                break;
            self.new_game()
            self.first_move()
            
            #print(self.column_counter)
            for i in range(41):
                self.move()
                
                if i >4:
                    if self.check_wins():
                        if self.current_turn=='p1':
                            self.p2_history['reward'][-1]=np.float32(1)
                            self.p1_history['reward'][-1]=np.float32(-1)
                            self.p2_history['next_state'].append(self.board.copy().astype(np.float32))
                            #print('p2_wins')
                        else:
                            self.p1_history['reward'][-1]=np.float32(1)
                            self.p2_history['reward'][-1]=np.float32(-1)
                            self.p1_history['next_state'].append(self.board.copy().astype(np.float32))
                            #print('p1_wins')
                        self.add_to_memory()
                
                        break;
        
        self.p1_memory=np.array(self.p1_memory).T
        self.p2_memory=np.array(self.p2_memory).T
        self.p1_memory=self.p1_memory[:n_of_turns,:]
        self.p2_memory=self.p2_memory[:n_of_turns,:]
        self.agent1.memory=self.p1_memory
        self.agent2.memory=self.p2_memory
        '''

        '''
        
            
    
#class Agent():
'''
class A():
    def __init__(self,agent):
        self.attribute=1
        self.agent=agent
        pass;
    def class_print(self):
        print('agents print function prints' +str(self.agent.attribute))

class B():
    def __init__(self):
        self.attribute=7
        pass;
    def class_print(self):
        print('this is class B print function')
'''