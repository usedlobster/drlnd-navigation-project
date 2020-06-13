from unityagents import UnityEnvironment

'''
    A very simple / basic class to abstract the UnityEnivronment even further
'''

class UnityEnvHelper:  

    # constructor - give file_name of agent environment

    def __init__( self , file_name  , no_graphics = True , seed = 8888 ):

        self.seed = seed
        self.uenv = UnityEnvironment( file_name = file_name , seed = self.seed , no_graphics = no_graphics )
       
        # pick the first agent as the brain
        
        self.brain_name = self.uenv.brain_names[0]
        self.brain = self.uenv.brains[self.brain_name]       
       
        # get the action space size 
  
        self.action_size = self.brain.vector_action_space_size
        
        # reset the environment , in training mode 
       
        self.reset( True  )
       
       # get the state space size 
        self.state_size = len( self.ue_info.vector_observations[0] )    
        
    def __del__(self):
        
        # make sure we close the environment 
        try:
            self.uenv.close() 
            del self.uenv
        except:
            pass
        
        
    def reset( self , train_mode = True  ):
        
        # tell the unity agent to restart an episode 
        # training mode simple seems to run the simulation at full speed 
        
        self.ue_info = self.uenv.reset( train_mode = train_mode )[self.brain_name]   
    # we pass in current state for convenience 
    def step( self , state ,  action ) :
       
        # perform action on environment  and get observation
        self.ue_info = self.uenv.step(action)[self.brain_name]   
        
        # return state , action and resulting state , reward and done flag 

        return { 'state':state,'action':action,'reward':self.reward(),'next_state':self.state(),'done':self.done() }
        
    def state( self ) :
        # just return what we think is current state 
        return self.ue_info.vector_observations[0]
        
    def reward( self ) :
        # return the reward from last step
        return self.ue_info.rewards[0]                   
    
    def done( self ) : 
        # return if done at last step 
        return self.ue_info.local_done[0]               

    

