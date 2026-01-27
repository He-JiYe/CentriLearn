<!-- # CentriLearn
The official implementation of CentriLearn: Learning to Identify Central Nodes in Complex Networks -->

<p align="center">
<h1 align="center"><strong>CentriLearn: Learning to Identify Central Nodes in Complex Networks</strong></h1>
<!-- <h3 align="center">CVPR XXX</h3> -->

## 项目描述
CentriLearn is an innovative reinforcement learning framework that solve the traditional complex network problems. More details can be found in our paper [PDF](网页)

## 项目架构
```
CENTRILEARN/
├─ config/                          # TODO:
│   └─ network_dismantling/         # 网络瓦解配置文件
│      ├─ dqn.yaml                  
│      └─ ppo.yaml
├─ ckpt/
│   └─ network_dismantling/         # 网络瓦解模型权重
│      ├─ dqn_100000.pth                  
│      └─ ppo_100000.pth
├─ data/                            # 真实网络
│   ├─ small/                       
│   └─ large/                       
├─ src/                             # 源代码
│  ├─ environments/                 # TODO: 环境
│  │  ├─ base.py                    
│  │  └─ network_dismantling.py     
│  ├─ models/                       # 模型
│  │  ├─ __init__.py                
│  │  ├─ nn/                        # 神经网络
│  │  │  ├─ __init__.py    
│  │  │  └─ GraphSAGE.py
│  │  ├─ backbones/                 # 骨干网络
│  │  │  ├─ __init__.py    
│  │  │  ├─ SimpleNet.py
│  │  │  ├─ DeepNet.py              
│  │  │  └─ FPNet.py                
│  │  ├─ network_dismantler/        # 网络瓦解模型
│  │  │  ├─ __init__.py    
│  │  │  ├─ Qnet.py                 
│  │  │  └─ ActorCritic.py          
│  │  └─ utils/                     # 模型工具
│  │     ├─ __init__.py    
│  │     ├─ registry.py
│  │     └─ builder.py
│  ├─ algorithms/                   # TODO:强化学习算法
│  │  ├─ base.py                   
│  │  ├─ dqn.py                    
│  │  └─ ppo.py                  
│  ├─ metrics/                      # TODO:评估指标  
│  └─ utils/                        # TODO:工具函数
├─ tools/                           # TODO:训练和测试脚本
│  └─ train.py                      
└─ README.md   
 ```