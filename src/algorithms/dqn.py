"""
Deep Q-Network (DQN) 算法实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Dict, Any, Optional, Tuple, Union
import random
import numpy as np
from .base import BaseAlgorithm
from src.utils import build_network_dismantler, build_replaybuffer, ALGORITHMS

@ALGORITHMS.register_module()
class DQN(BaseAlgorithm):
    """Deep Q-Network 算法

    实现带经验回放和目标网络的 DQN 算法。

    Args:
        model: Q-network 模型实例 或 模型配置字典
        optimizer_cfg: 优化器配置
        scheduler_cfg: 学习率调度器配置
        replay_buffer_cfg: 经验回放缓冲区配置
        dqn_cfg: DQN算法配置
        device: 运行设备
    """

    def __init__(self,
                 model: Union[nn.Module, Dict[str, Any]],
                 optimizer_cfg: Optional[Dict[str, Any]] = None,
                 scheduler_cfg: Optional[Dict[str, Any]] = None,
                 replaybuffer_cfg: Optional[Dict[str, Any]] = None,
                 algo_cfg: Optional[Dict[str, Any]] = None,
                 device: str = 'cpu'):
        """初始化 DQN 算法"""
        # 超参数
        self.gamma = algo_cfg.get('gamma', 0.99)
        self.epsilon_start = algo_cfg.get('epsilon_start', 1.0)
        self.epsilon_end = algo_cfg.get('epsilon_end', 0.01)
        self.epsilon_decay = algo_cfg.get('epsilon_decay', 10000)
        self.tau = algo_cfg.get('tau', 0.005)
        
        # 调用父类初始化（构建主模型）
        super().__init__(model, optimizer_cfg, scheduler_cfg, device)

        # 构建目标网络
        if isinstance(model, nn.Module):
            self.target_model = model.clone().to(device)
        elif isinstance(model, dict):
            self.target_model = self._build_model(model).to(device)
        else:
            raise TypeError(f"model 必须是 nn.Module 或 Dict，当前类型: {type(model)}")

        self.target_model.eval()
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 经验回放缓冲区
        self.replay_buffer = build_replaybuffer(replaybuffer_cfg)
    
    def _build_model(self, model_cfg: Dict[str, Any]) -> nn.Module:
        """从配置构建模型

        Args:
            model_cfg: 模型配置字典

        Returns:
            构建好的模型实例
        """
        return build_network_dismantler(model_cfg)

    def compute_epsilon(self) -> float:
        """计算当前探索率"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.training_step / self.epsilon_decay)
        return epsilon
    
    def select_action(self,
                      state: Dict[str, Any],
                      epsilon: Optional[float] = None) -> Tuple[torch.Tensor, float]:
        """选择动作（epsilon-greedy 策略）
        
        Args:
            state: 当前状态
            epsilon: 探索率（None 则自动计算）
            
        Returns:
            action: 选择的动作
            epsilon: 使用的探索率
        """
        if epsilon is None:
            epsilon = self.compute_epsilon()
        
        # Epsilon-greedy 策略
        if random.random() < epsilon:
            num_nodes = state['pyg_data'].x.shape[0]
            action = torch.randint(0, num_nodes, (1,))
        else:
            with torch.no_grad():
                info = state['pyg_data'].to(self.device)
                output = self.model({
                    'x': info.x,
                    'edge_index': info.edge_index,
                    'batch': info.get('batch', torch.zeros(info.x.shape[0], dtype=torch.long)),
                    'component': info.get('component'),
                })
                q_values = output['q_values'].squeeze(-1)
                action = torch.argmax(q_values)
        
        return action, epsilon
    
    def update(self, batch_size: int) -> Dict[str, float]:
        """更新模型

        Args:
            batch_size: 批次大小

        Returns:
            训练指标字典
        """
        if len(self.replay_buffer) < batch_size:
            return {'loss': 0.0}

        batch, indices, weights = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 前向传播
        self.set_train_mode()
        
        # 计算当前 Q 值
        state_info = Batch.from_data_list([state['pyg_data'] for state in states]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_state_info = Batch.from_data_list([next_state['pyg_data'] for next_state in next_states]).to(self.device)
        dones = torch.LongTensor(dones).to(self.device)

        # 当前 Q 值
        with torch.set_grad_enabled(True):
            output = self.model({
                    'x': state_info.x,
                    'edge_index': state_info.edge_index,
                    'batch': state_info.get('batch', torch.zeros(state_info.x.shape[0], dtype=torch.long)),
                    'component': state_info.get('component'),
                })
            current_q_values = output['q_values'].squeeze(-1)[actions]

        # 目标 Q 值
        with torch.no_grad():
            temp_value = self.model({
                    'x': next_state_info.x,
                    'edge_index': next_state_info.edge_index,
                    'batch': next_state_info.get('batch', torch.zeros(next_state_info.x.shape[0], dtype=torch.long)),
                    'component': next_state_info.get('component'),
            })
            next_actions = temp_value['q_values'].squeeze(-1).argmax(dim=0)

            next_output = self.target_model({
                    'x': next_state_info.x,
                    'edge_index': next_state_info.edge_index,
                    'batch': next_state_info.get('batch', torch.zeros(next_state_info.x.shape[0], dtype=torch.long)),
                    'component': next_state_info.get('component'),
            })
            next_q_values = next_output['q_values'].squeeze(-1)[next_actions]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)

        # 如果使用优先度采样，应用重要性采样权重
        if weights is not None:
            weights_tensor = torch.FloatTensor(weights).to(self.device)
            loss = (loss * weights_tensor).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新优先度（如果使用优先度采样）
        if indices is not None and weights is not None:
            with torch.no_grad():
                td_errors = torch.abs(current_q_values - target_q_values)
                priorities = td_errors.cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)
        
        # 更新目标网络
        self._update_target_network()
        
        self.training_step += 1
        
        return {
            'loss': loss.item(),
            'epsilon': self.compute_epsilon(),
            'training_step': self.training_step
        }
    
    def _update_target_network(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """执行一步训练（兼容基类接口）
        
        Args:
            batch: 训练数据批次
            
        Returns:
            训练指标字典
        """
        return self.update(batch.get('batch_size', 32))
    
    def collect_experience(self,
                           state: Dict[str, Any],
                           action: int,
                           reward: float,
                           next_state: Dict[str, Any],
                           done: bool):
        """收集经验到回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def get_q_values(self, state: Dict[str, Any]) -> torch.Tensor:
        """获取 Q 值

        Args:
            state: 状态

        Returns:
            Q 值
        """
        self.set_eval_mode()
        with torch.no_grad():
            info = state['pyg_data']
            output = self.model({
                'x': info.x,
                'edge_index': info.edge_index,
                'batch': info.get('batch', torch.zeros(info.x.shape[0], dtype=torch.long)),
                'component': info.component
            })
            return output['q_values'].squeeze(-1)

    def _run_training_loop(self,
                               env: Any,
                               training_cfg: Dict[str, Any],
                               verbose: bool = True) -> Dict[str, Any]:
        """DQN 训练循环实现

        Args:
            env: 环境实例
            training_cfg: 训练配置
            verbose: 是否打印日志

        Returns:
            训练结果字典
        """
        # 获取训练参数
        batch_size = training_cfg.get('batch_size', 32)
        update_freq = training_cfg.get('update_freq', 4)
        num_steps = training_cfg.get('num_steps', 10000)

        # 初始化
        state = env.reset()
        total_reward = 0

        if verbose:
            print("\n" + "=" * 60)
            print("开始 DQN 训练...")
            print("=" * 60)

        for step in range(num_steps):
            # 选择动作
            action, epsilon = self.select_action(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action, state['mapping'])
            total_reward += reward

            # 收集经验
            self.collect_experience(state, action, reward, next_state, done)

            # 更新状态
            state = next_state if not done else env.reset()

            # 更新模型
            if step % update_freq == 0 and len(self.replay_buffer) >= batch_size:
                metrics = self.update(batch_size)
                self.step_scheduler()

                # 打印训练日志
                if verbose and step % 100 == 0:
                    print(f"Step {step:6d} | Epsilon: {epsilon:.3f} | "
                        f"Loss: {metrics['loss']:.4f} | LR: {self.get_lr():.6f} | "
                        f"Avg Reward: {total_reward/(step+1):.4f}")

        return {
            'total_steps': num_steps,
            'total_reward': total_reward,
            'avg_reward': total_reward / num_steps,
            'final_lr': self.get_lr()
        }