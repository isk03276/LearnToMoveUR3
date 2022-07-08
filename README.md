# UR3-Deep-Reinforcement-Learning

You will be able to learn various tasks of the UR3 with robotiq85 gripper robot.  
Learning method is based on the DRL(Deep Reinforcement Learning).  
In this repo, we use [CoppeliaSim](http://www.coppeliarobotics.com/) (previously called V-REP), [Pyrep](https://github.com/stepjam/PyRep).  
Tasks  
 - Robotic manipulation tasks.
  - Reach target.
  - TO DO
 - TO DO

## Install
We assume python >= 3.7.9  
#### Coppeliasim
PyRep requires version **4.1(other versions may have bugs)** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)  
```bash
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

#### PyRep
Once you have downloaded and set CoppeliaSim, you can install PyRep:
Move to home workspace
```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install -e .
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.


#### LearnToMoveUR3
Move to home workspace
Clone repo and Install the python library:
```bash
git clone https://github.com/isk03276/LearnToMoveUR3.git
cd LearnToMoveUR3
pip install -r requirements.txt
```


## Getting Started
```bash
python main.py --env-id ENV_ID --load-from MODEL_CHECKPOINT_PATH #Train
python main.py --env-id reach --test --load-from MODEL_CHECKPOINT_PATH #Test
```


## Use Pretrained Model
```bash
python main.py --env-id reach --load-from pretrained_models/reach --test
```

