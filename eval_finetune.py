import os
import sys
import warnings
import torch
import torch.nn as nn
import hydra
import numpy as np
from termcolor import colored
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from common.logger import Logger

def smart_load_checkpoint(agent, checkpoint_path):
    print(colored(f'Loading weights for evaluation from {checkpoint_path}...', 'cyan', attrs=['bold']))
    
    try:
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = loaded['model'] if (isinstance(loaded, dict) and 'model' in loaded) else loaded
        if hasattr(state_dict, 'state_dict'): state_dict = state_dict.state_dict()
        if hasattr(state_dict, "to_dict"): state_dict = state_dict.to_dict()
        
        state_dict = {k: v for k, v in state_dict.items() if k != "__batch_size"}
        model_dict = agent.model.state_dict()
        
        compatible_weights = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if not isinstance(model_dict[k], torch.Tensor):
                    continue

                target_shape = model_dict[k].shape
                
                if v.shape == target_shape:
                    compatible_weights[k] = v
                elif len(v.shape) == len(target_shape):
                    print(colored(f"  Adapting layer {k}: {v.shape} -> {target_shape}", "yellow"))
                    new_weight = torch.empty_like(model_dict[k])
                    
                    if new_weight.dim() >= 2:
                        nn.init.orthogonal_(new_weight) 
                    else:
                        nn.init.zeros_(new_weight)
                    
                    slices = [slice(None)] * v.dim()
                    for d in range(v.dim()):
                        if v.shape[d] != target_shape[d]:
                            min_dim = min(v.shape[d], target_shape[d])
                            slices[d] = slice(0, min_dim)
                    
                    new_weight[tuple(slices)] = v[tuple(slices)]
                    compatible_weights[k] = new_weight
        
        model_dict.update(compatible_weights)
        agent.model.load_state_dict(model_dict)
        print(colored("Weights adapted successfully.", "green"))

    except Exception as e:
        print(colored(f"Load failed: {e}", "red"))
        import traceback
        traceback.print_exc()
        raise e

@hydra.main(config_name='config', config_path='.')
def evaluate_transfer(cfg: dict):
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored(f"Evaluating on task: {cfg.task}", "green"))

    env = make_env(cfg)
    agent = TDMPC2(cfg)
    logger = Logger(cfg)

    if cfg.checkpoint:
        smart_load_checkpoint(agent, cfg.checkpoint)
    else:
        print(colored("No checkpoint provided!", "red"))
        return

    rewards = []
    for i in range(cfg.eval_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        
        if cfg.save_video:
            logger.video.init(env, enabled=True)

        while not done:
            action = agent.act(obs, t0=t==0, eval_mode=True, task=None)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            t += 1
            if cfg.save_video: logger.video.record(env)
            
        rewards.append(ep_reward)
        if cfg.save_video: logger.video.save(step=i)
        print(f"Episode {i+1}: Reward {ep_reward:.1f}")

    print(f"Mean Reward over {cfg.eval_episodes} episodes: {np.mean(rewards):.1f}")

if __name__ == '__main__':
    evaluate_transfer()
