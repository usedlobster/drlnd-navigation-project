
from envhelper import UnityEnvHelper
from qnet import QNetwork
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 
import random,os

class TheAgent:

    def __init__( self , filename ,  seed = 12345 ):
        
        self.seed = seed 
        self.agent_file = filename 
        self.env = None
        self.q1 = None 
       
    def __del__( self ):

        self.end() 
    
    def start(self, viewer = False ):
    
        if self.env == None:
            self.env = UnityEnvHelper( self.agent_file , seed = self.seed , no_graphics = not viewer  )

        
    def end( self ):
        
        if self.env != None:
            del self.env
            self.env = None


    def save_model( self , filename = 'model.pt' ):
        
        try:
            if self.q1 != None:
                torch.save( self.q1.state_dict(), filename )
                print( f'Model saved as {filename}')
            else:
                raise 
        except:
            print( 'Failed to save model' )


    def load_model( self , filename = 'model.pt' ):

        if self.q1 != None:
                if os.path.exists( filename ):
                    self.q1.load_state_dict( torch.load( filename ))
                else:
                    print( f'\n model not found {filename}' )   
            

    def train( self , hyper_params , max_episodes = 2000 , goal = 13.0 , model_name = 'model.pt' ):
    
        # check if cuda device available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # save the hyperparmeters
        self.H = hyper_params 
        

        # create a fully connect neural network 
        self.q1 = QNetwork( self.env.state_size , self.H.FC1 , self.H.FC2 , self.env.action_size ).to(device)
        # randomize the weights 
        self.q1.randomize_weights()

        # create a second for traing 
        self.q2 = QNetwork( self.env.state_size , self.H.FC1 , self.H.FC2 , self.env.action_size ).to(device)
        # transfer weights from q1
        self.q2.copy_weights_from( self.q1 , 1.0 )

        # setup the ADAM  optimizer for training q1 
        optimizer = torch.optim.Adam( self.q1.parameters() , lr = self.H.ALPHA ) 
        # create replay buffer of given size
        replay_buffer = deque( maxlen= self.H.REPLAY_BUFFER_SIZE )

        # keep track of how many in buffer could use len( replay_buffer ) 
        n_replay = 0 

        # set epsilon to start value 
        eps = self.H.EPS_START

        # keep the scores for each episode here !
        self.scores = [] 

        for ith in range( 1 , max_episodes+1 ):

            # reset environment to start state

            self.env.reset( True ) 

            # score starts at 0 

            score = 0
            # get initial state vector
            state = self.env.state()

            while True:

                # pick action to explore / follow  using greedy+ 

                if ( score > 2 ) or ( random.random() > eps ):

                    # make state a tensor input
                    inp_v = torch.from_numpy(state).float().unsqueeze(0).to( device )                    
                    # get output for this state without training 

                    self.q1.eval()
                    with torch.no_grad():
                        out_v = self.q1( inp_v ).detach()
                    self.q1.train()

                    # get action index with maximum value 
                    action = np.argmax( out_v.cpu().data.numpy())

                else :                              

                    # just pick an action at pseudo random equaly likely                            
                    action = np.random.randint(self.env.action_size)

                # perform action - and get observation tuple ( including passed state / action )
                obs = self.env.step( state , action ) 

                # update the score
                score += obs['reward']
                # add observation to replay_buffer
                replay_buffer.append( obs )
                # keep track  of number 
                n_replay+=1 

                # train if we have at least 2xH.MINIBATCH_SIZE samples , every H.UPDATE_EVERY frames
                if ( n_replay > self.H.MINIBATCH_SIZE*2 ) and (( n_replay % self.H.UPDATE_EVERY ) == 0):
                    
                    # pick samples from replay_buffer with uniform probability 
                    samples = random.sample( replay_buffer , self.H.MINIBATCH_SIZE )

                    # extract to seperate state , next_state , reward , action , done vectors
                    # for batch processing 

                    b_s = torch.from_numpy(np.vstack([[d['state'] for d in samples ]])).float().to(device)
                    b_n = torch.from_numpy(np.vstack([[d['next_state'] for d in samples ]])).float().to(device)
                    b_r = torch.from_numpy(np.stack([[d['reward'] for d in samples ]],axis=-1)).float().to(device)
                    b_a = torch.from_numpy(np.stack([[d['action'] for d in samples ]],axis=-1)).long().to(device)
                    b_d = torch.from_numpy(np.stack([[d['done'] for d in samples ]] ,axis=-1).astype(np.uint8)).float().to(device)

                    # use q2 network to calcualte current 
                    Q_max = self.q2( b_n ).max(1)[0].unsqueeze(1)
                    Q_target = b_r + ( self.H.GAMMA * Q_max * ( 1 - b_d ))  

                    #
                    Q_exp = self.q1( b_s ).gather( 1 , b_a )  

                    # very important! - reset previous optimizer gradients values
                    optimizer.zero_grad()                  
                    # compute loss 
                    loss = F.mse_loss( Q_exp , Q_target ) 
                    # propagate loss backward 
                    loss.backward()

                    # run optimizer - this updates q1 - weights
                    optimizer.step()

                    # now transfer pro-rata weights from q1 to q2 
                    # as H.TAU is small , q2 is only changed very slowly 
                    # 
                    self.q2.copy_weights_from( self.q1 , self.H.TAU )
                
                # next frame of episode 

                if obs['done']:
                    break   

                state = obs['next_state']

            # save score            
            self.scores.append( score ) 
            # calculate the mean of the last 100 entries
            mean_score = np.mean( self.scores[-100:]) 

            if ( mean_score >= goal ) and ( ith >=100 ):
                print( f"\nSolved after {ith-100} episodes with mean score of = {mean_score:-6.3f} after 100 episodes")
                self.save_model( model_name )
                return True

            # update current score ( and newline every 100 )
            print ( f"\r{ith:5d} ,  {mean_score:6.3f}" , end='' if ( ith%100 ) else '\n'  )

            # decay epsilon 
            eps = max( self.H.EPS_END , self.H.EPS_DECAY * eps )

        print( f"\nModel not solved after {ith} episodes ")
        return False
            

    def play( self , hyper_params , no = 1  , train_mode = False , model_name = 'model.pt' ):
        #
        self.H = hyper_params 
        # create a network for training 
        self.q1 = QNetwork( self.env.state_size , self.H.FC1 , self.H.FC2 , self.env.action_size ) 
        #
        self.load_model( model_name )
        #
        self.scores = [] 
        #
        for ith in range( 1 , no  + 1 ):
            # reset environment to its initial state for this episode
            self.env.reset( train_mode ) 
            # score starts at 0 
            score = 0
            # get initial state 
            state = self.env.state()

            while True:

                inp_v = torch.from_numpy(state).float().unsqueeze(0)                    
                # get output for this state without training 
                out_v = self.q1( inp_v ).detach()
                # get action index with maximum value 
                action = np.argmax( out_v.data.numpy())
                # perform action - and get observation tuple ( including passed state / action ) 
                obs = self.env.step( state , action ) 
                # update the score
                score += obs['reward']
                # are we terminal 
                if obs['done']:
                    break

                state = obs['next_state']

            # save score            
            self.scores.append( score ) 
