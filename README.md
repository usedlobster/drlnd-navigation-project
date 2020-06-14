# Udacity  Deep Reinforcement Learning


## Project 1 - Navigation

[![A Trained agent](https://img.youtube.com/vi/AiWIgRDte3Q/default.jpg)](https://www.youtube.com/watch?v=AiWIgRDte3Q)

### Project Details 

For this project, we have been given the task of training an agent to navigate (and collect bananas!) in a large, square world. The world is an provided as a virtual environment using Unity Machine Learning Agents ( https://github.com/Unity-Technologies/ml-agents).

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:

- **0** - move forward.
- **1** - move backward.
- **2** - turn left.
- **3** - turn right.

The task is episodic and terminates after a fixed amount of time. The agent/task is solved when it can score an average of at least 13, over 100 consecutive episodes.

The agent code is required to be written in PyTorch and Python 3. It interacts with a custom provided unity app, that has an interface using the open-source Unity plugin (ML-Agents). For this project it is using ML-Agents version 0.4.0.

### Getting Started

1.  Clone or (download and unzip)  this repository.

2.  This project requires certain, library dependencies to be consistent - in particular the use of python >=3.6 , and a specific version of the ml_agents library version 0.4.0. These dependencies can be found in the pip requirements.txt file. 

    Different systems will vary, but an example configuration for setting up a conda environment  can be made as follows:-

    ```
    # create a conda enviroment 
    $ conda create --name drlnd python=3.6
    $ conda activate drlnd
    # install python dependencies 
    $ pip -r requirements.txt
    # install jupyter notebook kernel 
    $ python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
    For more details see [here](https://github.com/udacity/deep-reinforcement-learning#dependencies)

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. Unzip (or decompress) the file, into the same folder this repository has been saved.

### Instructions

The training code is found in the *agent.py* file , along with helper functions *envhelper.py* , and *qnet.py*. 

For convenince the interface to these is contained in the *Trainer.ipynb* jupyter / ipython notebook, where one can experemint and visualize the various HyperParameter choices. 

You need to run all the code in  ***section 0*** first , but can skip any of sections 1,2, and 3.

**0 . Setup **

Firstly, update the variable AGENT_FILE to match the path where you downloaded  the binary unity agent code.

If desired to train with new hyperparameters just change them in the HYPER_PARAMS class object.

Then execute all the cells in that section.

**1. Training** 

Training can be done , by simply executing the code in this section. The following parameters can be adjusted.

​	*seed* :  Start everything with this seed ( Agent/ torchj / numpy ).

​	*viewer* : set True to view environment

​	*max_episodes*: How many episodes to train for , before given up if not solved.

​	*model_name*: Where to save the trained model weights.

A plot of scores obtained during training is produced.

**2. Validation** 

Just to prove the model, we can load the agent again ( with different seed ) , and run another 100 episodes, and visualize the results.

This is usefull todo as its possible the model may have worsened during the later stages of training. ( see my Report.MD - for more information).

**3. Play **

For completness, it is possible to play a single episode with the trained model weights. By default with the viewer / at normal speed - but can be changed. It will print out the score at the end of the run.




