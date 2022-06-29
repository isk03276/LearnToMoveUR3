# CoppeliaSimUR3

You will be able to learn various tasks of the UR3 with robotiq85 gripper robot.  
Learning method is based on the DRL(Deep Reinforcement Learning).  
In this repo, we use [CoppeliaSim](http://www.coppeliarobotics.com/) (previously called V-REP), [Pyrep](https://github.com/stepjam/PyRep).  
Tasks  
 - Robotic manipulation tasks.
  - Reach target.
  - TO DO
 - TO DO

## Install

#### Coppeliasim
PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

We use python script in the coppeliaSim for manipulate robot, objects, etc.
So you have to set python path in the CoppeliaSim
```bash
cd EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR/system
gedit usrset.txt
```
Find "defaultPython = " then fill in your python path.
(ex. in my case -> defaultPython = "/home/eunjin/anaconda3/envs/ur3/bin/python3")

#### PyRep

Once you have downloaded and set CoppeliaSim, you can install PyRep:

```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install -e .
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

__Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.


#### Requirements
Finally install the python library:
```bash
pip install -r requirements.txt
```


## Getting Started



## Usage

