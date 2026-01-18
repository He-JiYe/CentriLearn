"""
PPO Training Example

This example demonstrates how to train a PPO agent on the Network Dismantling task
using the configuration file configs/network_dismantling/ppo.yaml.
"""

import torch
import yaml

from centrilearn.utils import (build_algorithm, build_environment,
                               train_from_cfg)


# Method 1: Training from config file (recommended)
def train_from_config_file():
    """Train PPO using the YAML config file."""
    print("=" * 60)
    print("Method 1: Training from config file")
    print("=" * 60)

    # Load configuration
    config_path = "configs/network_dismantling/ppo.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Modify training parameters for quick demonstration
    config["training"]["num_episodes"] = 100  # Reduce for quick demo
    config["training"]["log_interval"] = 10
    config["training"]["eval_interval"] = 50

    # Train
    print("\nStarting training...")
    results, algorithm = train_from_cfg(config, verbose=True)

    # Print results
    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    print(f"Total episodes: {results.get('total_episodes', 0)}")
    print(f"Average reward: {results.get('avg_reward', 0):.4f}")
    print(f"Best reward: {results.get('best_reward', 0):.4f}")

    # Save the trained model
    save_path = "ckpt/network_dismantling/ppo_trained.pth"
    algorithm.save_checkpoint(save_path, episode=results.get("total_episodes", 0))
    print(f"\nModel saved to: {save_path}")

    return algorithm, results


# Method 2: Training with custom configuration
def train_with_custom_config():
    """Train PPO with custom configuration."""
    print("\n" + "=" * 60)
    print("Method 2: Training with custom configuration")
    print("=" * 60)

    # Define custom configuration
    config = {
        "algorithm": {
            "type": "PPO",
            "model": {
                "type": "ActorCritic",
                "backbone_cfg": {
                    "type": "GraphSAGE",
                    "in_channels": 2,
                    "hidden_channels": 64,
                    "num_layers": 3,
                },
                "actor_head_cfg": {"type": "PolicyHead", "in_channels": 64},
                "critic_head_cfg": {"type": "VHead", "in_channels": 64},
            },
            "optimizer_cfg": {"type": "Adam", "lr": 0.0001, "weight_decay": 0.0005},
            "replaybuffer_cfg": {"type": "RolloutBuffer", "capacity": 2048},
            "metric_manager_cfg": {
                "save_dir": "./logs/metrics",
                "log_interval": 10,
                "metrics": [
                    {"type": "AUC", "record": "min"},
                    {"type": "AttackRate", "record": "min"},
                ],
            },
            "algo_cfg": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "max_grad_norm": 0.5,
                "num_epochs": 10,
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "environment": {
            "type": "NetworkDismantlingEnv",
            "synth_type": "ba",
            "synth_args": {"min_n": 30, "max_n": 50, "m": 4},
            "node_features": "combin",
            "is_undirected": True,
            "value_type": "ar",
            "use_gcc": False,
            "use_component": False,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "training": {
            "num_episodes": 100,
            "max_steps": 1000,
            "batch_size": 64,
            "log_interval": 10,
            "eval_interval": 50,
            "eval_episodes": 1,
        },
    }

    # Train
    print("\nStarting training...")
    results, algorithm = train_from_cfg(config, verbose=True)

    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    print(f"Total episodes: {results.get('total_episodes', 0)}")
    print(f"Average reward: {results.get('avg_reward', 0):.4f}")

    return algorithm, results


# Method 3: Step-by-step training
def train_step_by_step():
    """Train PPO step by step for more control."""
    print("\n" + "=" * 60)
    print("Method 3: Step-by-step training")
    print("=" * 60)

    # Build environment
    env_cfg = {
        "type": "NetworkDismantlingEnv",
        "synth_type": "ba",
        "synth_args": {"n": 50, "m": 2},
        "node_features": "combin",
        "value_type": "ar",
    }
    env = build_environment(env_cfg)

    # Build algorithm
    algo_cfg = {
        "type": "PPO",
        "model": {
            "type": "ActorCritic",
            "backbone_cfg": {
                "type": "GraphSAGE",
                "in_channels": 2,
                "hidden_channels": 64,
                "num_layers": 3,
            },
            "actor_head_cfg": {"type": "PolicyHead", "in_channels": 64},
            "critic_head_cfg": {"type": "VHead", "in_channels": 64},
        },
        "optimizer_cfg": {"type": "Adam", "lr": 0.0001},
        "replaybuffer_cfg": {"type": "RolloutBuffer", "capacity": 2048},
        "algo_cfg": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "num_epochs": 10,
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    algorithm = build_algorithm(algo_cfg)

    # Training loop
    num_episodes = 50
    batch_size = 64
    episode_rewards = []

    algorithm.set_train_mode()

    print("\nStarting training loop...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            # Select action
            action, log_prob, value = algorithm.select_action(state)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Collect experience
            algorithm.collect_experience(state, action, log_prob, reward, done, value)

            state = next_state
            episode_reward += reward
            step += 1

        episode_rewards.append(episode_reward)

        # Update model at the end of episode
        if len(algorithm.replay_buffer) > 0:
            loss_info = algorithm.update(batch_size)
            if (episode + 1) % 10 == 0:
                print(
                    f"  Update - Policy Loss: {loss_info.get('policy_loss', 0):.4f}, "
                    f"Value Loss: {loss_info.get('value_loss', 0):.4f}, "
                    f"Entropy: {loss_info.get('entropy_loss', 0):.4f}"
                )

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Steps: {step}, "
                f"Reward: {episode_reward:.4f}, "
                f"Avg Reward (last 10): {avg_reward:.4f}"
            )

    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    print(f"Total episodes: {num_episodes}")
    print(f"Average reward: {sum(episode_rewards) / num_episodes:.4f}")
    print(f"Best reward: {max(episode_rewards):.4f}")

    return algorithm, episode_rewards


# Method 4: Evaluation
def evaluate_trained_model(checkpoint_path="ckpt/network_dismantling/ppo_trained.pth"):
    """Evaluate a trained PPO model."""
    print("\n" + "=" * 60)
    print("Method 4: Evaluating trained model")
    print("=" * 60)

    # Build environment
    env_cfg = {
        "type": "NetworkDismantlingEnv",
        "synth_type": "ba",
        "synth_args": {"n": 50, "m": 2},
        "node_features": "combin",
        "value_type": "ar",
    }
    env = build_environment(env_cfg)

    # Build algorithm
    algo_cfg = {
        "type": "PPO",
        "model": {
            "type": "ActorCritic",
            "backbone_cfg": {
                "type": "GraphSAGE",
                "in_channels": 2,
                "hidden_channels": 64,
                "num_layers": 3,
            },
            "actor_head_cfg": {"type": "PolicyHead", "in_channels": 64},
            "critic_head_cfg": {"type": "VHead", "in_channels": 64},
        },
        "optimizer_cfg": {"type": "Adam", "lr": 0.0001},
        "algo_cfg": {"gamma": 0.99, "gae_lambda": 0.95},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    algorithm = build_algorithm(algo_cfg)

    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = algorithm.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint from episode: {checkpoint.get('episode', 'unknown')}")

    # Set to evaluation mode
    algorithm.set_eval_mode()

    # Evaluate
    num_episodes = 10
    episode_rewards = []

    print("\nEvaluating...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action (deterministic)
            action, value = algorithm.get_action_value(state)

            # Execute action
            next_state, reward, done, info = env.step(action)

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(
            f"Episode {episode + 1}/{num_episodes}, "
            f"Reward: {episode_reward:.4f}, "
            f"Steps: {info.get('steps', 'N/A')}"
        )

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"Average reward: {sum(episode_rewards) / num_episodes:.4f}")
    print(f"Best reward: {max(episode_rewards):.4f}")
    print(f"Worst reward: {min(episode_rewards):.4f}")

    return episode_rewards


# Method 5: Using VectorizedEnv for faster training
def train_with_vectorized_env():
    """Train PPO using vectorized environment for parallel training."""
    print("\n" + "=" * 60)
    print("Method 5: Training with VectorizedEnv")
    print("=" * 60)

    # Build vectorized environment
    env_cfg = {
        "type": "NetworkDismantlingEnv",
        "synth_type": "ba",
        "synth_args": {"n": 50, "m": 2},
        "node_features": "combin",
        "value_type": "ar",
        "env_num": 4,  # Create 4 parallel environments
    }
    env = build_environment(env_cfg)

    # Build algorithm
    algo_cfg = {
        "type": "PPO",
        "model": {
            "type": "ActorCritic",
            "backbone_cfg": {
                "type": "GraphSAGE",
                "in_channels": 2,
                "hidden_channels": 64,
                "num_layers": 3,
            },
            "actor_head_cfg": {"type": "PolicyHead", "in_channels": 64},
            "critic_head_cfg": {"type": "VHead", "in_channels": 64},
        },
        "optimizer_cfg": {"type": "Adam", "lr": 0.0001},
        "replaybuffer_cfg": {"type": "RolloutBuffer", "capacity": 2048},
        "algo_cfg": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "num_epochs": 10,
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    algorithm = build_algorithm(algo_cfg)

    # Training loop with vectorized environment
    num_episodes = 50
    batch_size = 64
    num_envs = len(env)
    episode_rewards = [[] for _ in range(num_envs)]

    algorithm.set_train_mode()

    print(f"\nTraining with {num_envs} parallel environments...")
    for episode in range(num_episodes):
        # Reset all environments
        states = env.reset()
        done_flags = [False] * num_envs
        step = 0

        # Run until all environments are done
        while not all(done_flags):
            # Select actions for all environments
            actions = []
            for i, state in enumerate(states):
                if done_flags[i]:
                    actions.append(0)
                else:
                    action, log_prob, value = algorithm.select_action(state)
                    actions.append(action)

            # Execute actions in all environments
            next_states, rewards, dones, infos = env.step(actions)

            # Collect experiences
            for i in range(num_envs):
                if not done_flags[i]:
                    state = states[i]
                    action = actions[i]
                    log_prob, value = (
                        algorithm.select_action(state)[1],
                        algorithm.select_action(state)[2],
                    )
                    reward = rewards[i]
                    done = dones[i]

                    algorithm.collect_experience(
                        state, action, log_prob, reward, done, value
                    )

                    if not done_flags[i]:
                        episode_rewards[i].append(reward)

                    if done:
                        done_flags[i] = True

            states = next_states
            step += 1

        # Update model
        if len(algorithm.replay_buffer) > 0:
            loss_info = algorithm.update(batch_size)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_rewards = [sum(r) for r in episode_rewards if r]
            if avg_rewards:
                avg_reward = sum(avg_rewards) / len(avg_rewards)
                print(
                    f"Episode {episode + 1}/{num_episodes}, "
                    f"Avg Reward: {avg_reward:.4f}"
                )

    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    all_rewards = [sum(r) for r in episode_rewards if r]
    if all_rewards:
        print(f"Average reward: {sum(all_rewards) / len(all_rewards):.4f}")
        print(f"Best reward: {max(all_rewards):.4f}")

    return algorithm, episode_rewards


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CentriLearn PPO Training Examples")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Run examples
    algorithm1, results1 = train_from_config_file()
    algorithm2, results2 = train_with_custom_config()
    algorithm3, rewards3 = train_step_by_step()
    algorithm4, rewards4 = train_with_vectorized_env()

    # Evaluate (if checkpoint exists)
    try:
        evaluate_trained_model()
    except Exception as e:
        print(f"\nEvaluation skipped: {e}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
