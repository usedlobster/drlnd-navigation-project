from agent import TheAgent

AGENT_FILE = '../../Banana_Linux/Banana.x86_64'


class HYPER_PARAMS(object):

    FC1 = 48
    FC2 = 48
    REPLAY_BUFFER_SIZE = 20000
    MINIBATCH_SIZE = 64
    UPDATE_EVERY = 10
    GAMMA = 0.995
    ALPHA = 0.001
    TAU = 0.01
    EPS_START = 1.0
    EPS_END   = 0.01
    EPS_DECAY = 0.99
    # make constant(ish)
    def __setattr__(self, *_):
        pass

try:
	
    agent = TheAgent( AGENT_FILE , HYPER_PARAMS  ) 
    agent.start( no_viewer = True) 
    agent.train() 

finally:
    agent.end() 
    del agent 
