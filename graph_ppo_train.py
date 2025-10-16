import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from model import Agent
from environment import CloudConsolidationEnv


@dataclass
class Args:
    """Training configuration arguments."""
    exp_name: str = os.path.basename(__file__)[:-3]
    """Name of this experiment"""
    seed: int = 42
    """Random seed for reproducibility"""
    torch_deterministic: bool = True
    """Use deterministic algorithms in PyTorch (if supported)"""
    cuda: bool = True
    """Enable CUDA if available"""
    track: bool = False
    """Log to Weights & Biases"""
    wandb_project_name: str = "cleanRL"
    """W&B project name"""
    wandb_entity: Optional[str] = None
    """W&B team/entity name"""
    capture_video: bool = False
    """Whether to record videos (saved in 'videos/' folder)"""

    # Environment and training parameters
    env_id: str = "CloudConsolidation-v0"
    """Gym environment ID"""
    total_timesteps: int = 300_000
    """Total number of environment steps to train"""
    learning_rate: float = 2.5e-4
    """Learning rate for Adam optimizer"""
    num_envs: int = 1
    """Number of parallel environments (currently fixed at 1)"""
    num_steps: int = 64
    """Number of steps per rollout before updating policy"""
    anneal_lr: bool = False
    """Linearly decay learning rate to zero over training"""
    gamma: float = 0.99
    """Discount factor for returns"""
    gae_lambda: float = 0.95
    """Lambda parameter for GAE (Generalized Advantage Estimation)"""
    num_minibatches: int = 1
    """Number of mini-batches per update epoch"""
    update_epochs: int = 4
    """Number of optimization epochs per policy update"""
    norm_adv: bool = False
    """Normalize advantages within each batch"""
    clip_coef: float = 0.2
    """PPO clipping coefficient for policy loss"""
    clip_vloss: bool = True
    """Use clipped value loss as in PPO paper"""
    ent_coef: float = 0.01
    """Entropy regularization coefficient"""
    vf_coef: float = 0.5
    """Value function loss coefficient"""
    max_grad_norm: float = 0.5
    """Maximum gradient norm for clipping"""
    target_kl: Optional[float] = None
    """Early stopping based on KL divergence threshold"""

    # Computed at runtime
    batch_size: int = 0
    """(Computed) Total rollout batch size = num_envs * num_steps"""
    minibatch_size: int = 0
    """(Computed) Size of each mini-batch"""
    num_iterations: int = 0
    """(Computed) Number of total training iterations"""

    # Training-specific settings
    save_model_interval: int = 10_000
    """Save model every N global steps"""
    model_name: Optional[str] = None
    """Optional pre-trained model checkpoint name (without extension)"""


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str):
    """
    Factory function to create a single environment with wrappers.

    Args:
        env_id: Gym environment ID
        idx: Environment index (for video recording)
        capture_video: Whether to record videos
        run_name: Unique identifier for logging

    Returns:
        Callable that creates the environment when invoked
    """

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def main():
    """Main training loop for PPO agent."""
    # Register custom environment
    gym.envs.registration.register(
        id='CloudConsolidation-v0',
        entry_point='envs.environment_fa_std:CloudConsolidationEnv'
    )

    # Parse command-line arguments
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Initialize logging
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{k}|{v}|" for k, v in vars(args).items())
    )

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create and configure environment
    env = gym.make('CloudConsolidation-v0')
    dataset_name = 'fa_traindata'
    select_list = []
    env.unwrapped.configure_env('train', dataset_name, do_select=True, select_list=select_list)

    # Initialize agent and optimizer
    agent = Agent(device).to(device)
    if args.model_name is not None:
        model_path = f"./checkpoint/{args.model_name}.pt"
        agent.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pre-trained model from {model_path}")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Buffer setup for rollout storage
    obs = np.empty((args.num_steps, args.num_envs), dtype=object)
    actions0 = torch.zeros((args.num_steps, args.num_envs), device=device)
    actions1 = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs0 = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs1 = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    terminations = torch.zeros((args.num_steps, args.num_envs), device=device)
    truncations = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    nextvalues = torch.zeros((args.num_steps, args.num_envs), device=device)
    edge_candidates = np.empty((args.num_steps, args.num_envs), dtype=object)

    # Training loop
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_done = False

    for iteration in range(1, args.num_iterations + 1):
        # Anneal learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        step = 0
        while step < args.num_steps:
            while not next_done:
                global_step += 1
                obs[step] = next_obs
                edge_candidates[step] = env.unwrapped.edge_candidates.copy()

                with torch.no_grad():
                    action, logprob, _ = agent.get_action([next_obs], edge_candidates[step])
                    value = agent.get_value(next_obs).flatten()

                actions0[step] = action[0].item()
                actions1[step] = action[1].item()
                logprobs0[step] = logprob[0].item()
                logprobs1[step] = logprob[1].item()
                values[step] = value.item()

                # Execute action
                action_np = action.view(-1, 2).cpu().numpy()
                next_obs, reward, terminated, truncated, infos = env.step(action_np)
                next_done = terminated or truncated

                rewards[step] = torch.tensor(reward, device=device)
                terminations[step] = terminated
                truncations[step] = truncated

                with torch.no_grad():
                    next_value = agent.get_value(next_obs).flatten()
                    nextvalues[step] = next_value.item()

                step += 1

                # Logging episode results
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}, "
                                  f"data_ptr={info['data_ptr']}, data_level={info['data_level']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        if 'unsolved_data_num' in info:
                            writer.add_scalar("charts/unsolved_data_num", info['unsolved_data_num'], global_step)
                        if 'data_level' in info:
                            writer.add_scalar("charts/data_level", info['data_level'], global_step)

        # Compute advantages using GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextval = next_value
                else:
                    nextnonterminal = 1.0 - terminations[t + 1]
                    nextval = values[t + 1]
                delta = rewards[t] + args.gamma * nextval * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch for optimization
        b_obs = obs.reshape(-1)
        b_actions0 = actions0.reshape(-1)
        b_actions1 = actions1.reshape(-1)
        b_logprobs0 = logprobs0.reshape(-1)
        b_logprobs1 = logprobs1.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_edge_candidates = edge_candidates.reshape(-1)

        # Optimize policy and value networks
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = agent.get_action(
                    [b_obs[i] for i in mb_inds],
                    [b_edge_candidates[i] for i in mb_inds],
                    (b_actions0[mb_inds].long(), b_actions1[mb_inds].long())
                )
                newvalue = agent.get_value([b_obs[i] for i in mb_inds]).view(-1)
                logratio = newlogprob - torch.stack((b_logprobs0[mb_inds], b_logprobs1[mb_inds]))
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean().item()
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                if args.clip_vloss:
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef,
                                                                args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * v_loss_unclipped.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(f"SPS: {int(global_step / (time.time() - start_time))}")

        # Save model checkpoint
        if global_step >= save_model_threshold:
            model_path = f"models/ppo_{run_name}_{global_step}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            save_model_threshold += args.save_model_interval

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
