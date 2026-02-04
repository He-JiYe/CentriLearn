"""
Proximal Policy Optimization (PPO) 算法实现
适用于网络瓦解等离散动作空间任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_scatter import scatter_log_softmax, scatter_softmax
from typing import Dict, Any, Optional, Tuple, Union
from .base import BaseAlgorithm
from src.utils import build_network_dismantler, ALGORITHMS


@ALGORITHMS.register_module()
class PPO(BaseAlgorithm):
    """Proximal Policy Optimization 算法

    实现 Actor-Critic 架构的 PPO-Clip 算法。

    Args:
        model: Actor-Critic 模型实例 或 模型配置字典
        optimizer_cfg: 优化器配置
        scheduler_cfg: 学习率调度器配置
        replaybuffer_cfg: 轨迹缓冲区配置
        metric_manager_cfg: 指标管理器配置
        algo_cfg: 算法配置
        device: 运行设备
    """

    def __init__(self,
                 model: Union[nn.Module, Dict[str, Any]],
                 optimizer_cfg: Optional[Dict[str, Any]] = None,
                 scheduler_cfg: Optional[Dict[str, Any]] = None,
                 replaybuffer_cfg: Optional[Dict[str, Any]] = None,
                 metric_manager_cfg: Optional[Dict[str, Any]] = None,
                 algo_cfg: Optional[Dict[str, Any]] = None,
                 device: str = 'cpu'):
        """初始化 PPO 算法"""
        # 超参数
        self.gamma = algo_cfg.get('gamma', 0.99)
        self.gae_lambda = algo_cfg.get('gae_lambda', 0.95)
        self.clip_epsilon = algo_cfg.get('clip_epsilon', 0.2)
        self.entropy_coef = algo_cfg.get('entropy_coef', 0.01)
        self.value_coef = algo_cfg.get('value_coef', 0.5)
        self.max_grad_norm = algo_cfg.get('max_grad_norm', 0.5)
        self.num_epochs = algo_cfg.get('num_epochs', 10)
        
        # 调用父类初始化（支持模型配置）
        super().__init__(model, optimizer_cfg, scheduler_cfg, replaybuffer_cfg, metric_manager_cfg, device)

    def _build_model(self, model_cfg: Dict[str, Any]) -> nn.Module:
        """从配置构建模型

        Args:
            model_cfg: 模型配置字典

        Returns:
            构建好的模型实例
        """
        return build_network_dismantler(model_cfg)
    
    def select_action(self,
                      state: Dict[str, Any],
                      deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """选择动作

        Args:
            state: 当前状态
            deterministic: 是否确定性策略

        Returns:
            action: 选择的动作
            log_prob: 动作对数概率
            value: 状态价值（标量）
        """
        self.set_eval_mode()

        with torch.no_grad():
            info = state['pyg_data']
            output = self.model({
                'x': info.x,
                'edge_index': info.edge_index,
                'batch': info.get('batch', torch.zeros(info.x.shape[0], dtype=torch.long, device=self.device)),
                'component': info.component,
            })

            logit = output['logit'].squeeze()
            value = output['v_values'].squeeze()

            if deterministic:
                # 确定性策略：选择概率最大的动作
                action = torch.argmax(logit, dim=0)
                log_prob = F.log_softmax(logit, dim=0)[action]
            else:
                # 随机策略：按概率采样
                probs = F.softmax(logit, dim=0)
                action = torch.multinomial(probs, 1)
                log_prob = F.log_softmax(logit, dim=0)[action]

        return action.squeeze(), log_prob.squeeze(), value
    
    def collect_experience(self,
                           state: Dict[str, Any],
                           action: torch.Tensor,
                           log_prob: torch.Tensor,
                           reward: float,
                           done: bool,
                           value: torch.Tensor):
        """收集经验到轨迹缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            log_prob: 动作对数概率
            reward: 获得的奖励
            done: 是否终止
            value: 状态价值估计
        """
        self.replay_buffer.push(state, action.item(), log_prob.item(), reward, done, value.item())
    
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """更新模型
        
        Args:
            batch_size: 批次大小
            
        Returns:
            训练指标字典
        """
        if len(self.replay_buffer) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
        
        # 获取训练批次
        batches = self.replay_buffer.get_batches(
            batch_size=batch_size,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        self.set_train_mode()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        num_updates = 0
        
        # 多轮更新
        for _ in range(self.num_epochs):
            for batch in batches:
                # 前向传播
                policy_loss_epoch = 0
                value_loss_epoch = 0
                entropy_loss_epoch = 0
    
                state_info = Batch.from_data_list([i['pyg_data'] for i in batch['states']]).to(self.device)
                actions = torch.LongTensor([batch['actions']]).to(self.device)
                old_log_probs = torch.FloatTensor([batch['old_log_probs']]).to(self.device)
                returns = torch.FloatTensor([batch['returns']]).to(self.device)
                advantages = torch.FloatTensor([batch['advantages']]).to(self.device)
                old_values = torch.FloatTensor([batch['old_values']]).to(self.device)

                # 前向传播
                batch_indices = state_info.get('batch', torch.zeros(state_info.x.shape[0], dtype=torch.long))
                output = self.model({
                    'x': state_info.x,
                    'edge_index': state_info.edge_index,
                    'batch': batch_indices,
                    'component': state_info.component,
                })

                new_logit = output['logit'].squeeze()           
                new_value = output['v_values'].squeeze()        

                # 计算 ratio
                log_prob = scatter_log_softmax(new_logit, batch_indices, dim=0) 
                new_log_prob = log_prob[actions]                            
                ratio = torch.exp(new_log_prob - old_log_probs)

                # 计算 PPO loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss_epoch = -torch.min(surr1, surr2).mean()

                # 计算 value loss
                value_clipped = old_values + torch.clamp(new_value - old_values, -self.clip_epsilon, self.clip_epsilon)
                value_loss1 = F.mse_loss(new_value.unsqueeze(0), returns, reduction='none')
                value_loss2 = F.mse_loss(value_clipped, returns, reduction='none')
                value_loss_epoch = torch.max(value_loss1, value_loss2).mean()

                # 计算熵损失
                probs = scatter_softmax(new_logit, batch_indices, dim=0)
                entropy_loss_epoch = -torch.sum(probs * log_prob)

                # 合并 losses
                policy_loss = policy_loss_epoch
                value_loss = self.value_coef * value_loss_epoch
                entropy_loss = -self.entropy_coef * entropy_loss_epoch

                total_loss = policy_loss + value_loss + entropy_loss

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1
        
        self.training_step += num_updates
        
        # 清空缓冲区
        self.replay_buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0.0,
            'value_loss': total_value_loss / num_updates if num_updates > 0 else 0.0,
            'entropy_loss': total_entropy_loss / num_updates if num_updates > 0 else 0.0,
            'training_step': self.training_step
        }
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """执行一步训练（兼容基类接口）
        
        Args:
            batch: 训练数据批次
            
        Returns:
            训练指标字典
        """
        return self.update(batch.get('batch_size', 64))
    
    def get_action_value(self, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作和价值（用于推理）

        Args:
            state: 状态

        Returns:
            action: 选择的动作
            value: 状态价值（标量）
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
            logit = output['logit'].squeeze()
            value = output['v_values'].squeeze()

            action = torch.argmax(logit, dim=0)
        return action, value

    def _run_training_loop(self,
                           env: Any,
                           training_cfg: Dict[str, Any],
                           verbose: bool = True) -> Dict[str, Any]:
        """PPO 训练循环实现

        Args:
            env: 环境实例
            training_cfg: 训练配置
            verbose: 是否打印日志

        Returns:
            训练结果字典
        """
        # 获取训练参数
        batch_size = training_cfg.get('batch_size', 64)
        max_steps = training_cfg.get('max_steps', 1000)
        num_episodes = training_cfg.get('num_episodes', 100)
        log_interval = training_cfg.get('log_interval', 10)
        eval_interval = training_cfg.get('eval_interval', 50)
        eval_episodes = training_cfg.get('eval_episodes', 5)

        # 初始化
        total_reward = 0
        episode_rewards = []

        if verbose:
            print("\n" + "=" * 60)
            print("开始 PPO 训练...")
            print("=" * 60)

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            for _ in range(max_steps):
                # 选择动作
                action, log_prob, value = self.select_action(state)

                # 执行动作
                next_state, reward, done, info = env.step(action, state['mapping'])
                episode_reward += reward

                # 收集经验
                self.collect_experience(state, action, log_prob, reward, done, value)

                if self.metric_manager is not None:
                    self.metric_manager.update(
                        state, action, reward, next_state, done,
                        {'lcc_size': env.lcc_size[-1], 'num_nodes': env.num_nodes}
                    )

                # 更新状态
                state = next_state if not done else env.reset()

                if done:
                    break

            # 更新模型
            if len(self.replay_buffer) > 0:
                metrics = self.update(batch_size)
                self.step_scheduler(metrics)

            total_reward += episode_reward
            episode_rewards.append(episode_reward)

            # 打印训练日志
            if verbose and (episode + 1) % log_interval == 0:
                avg_reward = sum(episode_rewards[-log_interval:]) / len(episode_rewards[-log_interval:])
                print(f"Episode {episode+1:4d} | Reward: {episode_reward:8.4f} | "
                    f"Avg Reward ({log_interval}): {avg_reward:8.4f} | LR: {self.get_lr():.6f}")

            # 定期评估
            if (episode + 1) % eval_interval == 0 and self.metric_manager is not None:
                if verbose:
                    print(f"\n  [评估 Episode {episode + 1}]")
                self.set_eval_mode()
                eval_results = self.metric_manager.evaluate(env, self, eval_episodes)
                if verbose:
                    for name, result in eval_results.items():
                        current = result.get('current', 0.0)
                        print(f"    {name}: {current:.4f}")
                self.set_train_mode()

        return {
            'total_episodes': num_episodes,
            'total_reward': total_reward,
            'avg_reward': total_reward / num_episodes,
            'final_lr': self.get_lr(),
            'metrics': self.metric_manager.get_results() if self.metric_manager else None
        }
