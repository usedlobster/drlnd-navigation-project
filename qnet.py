import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    

    def __init__(self, state_size, fc1_size , fc2_size , action_size , seed = 1234 ):
        
        """
        
        Build the NN Model with a series of linear transformation layers, 
        with the layer sizes given by the parameters above.
        
        state_size > fc1_size > fc2_size > action_size 
            
        """
        
        super(QNetwork, self).__init__()
        
        if seed != None :
            torch.manual_seed( seed )
            
        self.fc1 = nn.Linear(state_size, fc1_size )
        self.fc2 = nn.Linear( fc1_size , fc2_size )
        self.fc3 = nn.Linear( fc2_size , action_size )
        
       
        
    def forward(self, input_vector ):
        
        # forward pass        
        x = F.relu(self.fc1(input_vector))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def randomize_weights( self ) :
        
        
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)
        
        nn.init.normal_(self.fc1.bias.data)
        nn.init.normal_(self.fc2.bias.data)
        nn.init.normal_(self.fc3.bias.data)
        
    
    
    def copy_weights_from( self , src_model , tau = 1.0 ):
        #
        # src_model must be a QNetwork of same type/size 
        #
        # update this instances weight/parameters from another QNetwork model
        # with proportion τ
        #
        # θ = τ*θ_src + (1 - τ)*θ_dst
        #
        # so τ=1.0 copies paremeters 100% from the source model 
        
        for src_p , dst_p in zip( src_model.parameters() , self.parameters()  ):
            dst_p.data.copy_(  tau * src_p.data + (1.0-tau)*dst_p.data  )    
    
   

