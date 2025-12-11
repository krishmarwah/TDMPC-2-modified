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
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def patched_update(self, buffer):
    batch = buffer.sample()
    
    obs, action, reward, next_obs, terminated, task = None, None, None, None, None, None
    if isinstance(batch, (tuple, list)):
        if len(batch) == 6:
            obs, action, reward, task, next_obs, terminated = batch
        elif len(batch) == 5:
            obs, action, reward, next_obs, terminated = batch
        else:
            obs, action, reward = batch[0], batch[1], batch[2]
            terminated = batch[-1]
    else:
        obs, action, reward = batch.get('obs'), batch.get('action'), batch.get('reward')
        terminated, task = batch.get('terminated'), batch.get('task')

    if terminated is None: terminated = torch.zeros_like(reward)
    
    task = None 

    try:
        return self._update(obs=obs, action=action, reward=reward, terminated=terminated, task=task)
    except TypeError:
        try:
             return self._update(obs, action, reward, terminated, task)
        except TypeError:
             return self._update(obs, action, reward, terminated)
    except AssertionError as e:
        raise e
    except AttributeError:
        return {"loss": 0.0}

TDMPC2.update = patched_update


def smart_load_checkpoint(agent, checkpoint_path, freeze_encoder=False):
    print(colored(f'Loading transfer weights from {checkpoint_path}...', 'cyan', attrs=['bold']))
    
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
                target_shape = model_dict[k].shape
                if v.shape == target_shape:
                    compatible_weights[k] = v
                elif len(v.shape) == len(target_shape):
                    print(colored(f"  Reshaping layer {k}: Checkpoint {v.shape} -> Current {target_shape}", "yellow"))
                    
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
        
        task_emb_keys = [k for k in state_dict.keys() if 'task_emb' in k]
        if task_emb_keys and 'task_emb' in model_dict:
            print(colored("  Initializing task embedding with mean of pre-trained tasks...", "yellow"))
            pt_embs = state_dict[task_emb_keys[0]] 
            mean_emb = pt_embs.mean(dim=0, keepdim=True)
            if model_dict['task_emb'].shape[0] > 0:
                 model_dict['task_emb'].data[0] = mean_emb.squeeze()
            
            if task_emb_keys[0] in compatible_weights:
                del compatible_weights[task_emb_keys[0]]

        model_dict.update(compatible_weights)
        agent.model.load_state_dict(model_dict)
        print(colored("Transfer weights loaded successfully.", "green"))

    except Exception as e:
        print(colored(f"CRITICAL LOAD ERROR: {e}", "red"))
        raise e

    if freeze_encoder:
        print(colored('Freezing encoder layers...', 'yellow'))
        enc = getattr(agent.model, 'encoder', getattr(agent.model, '_encoder', None))
        if not enc:
            for n, m in agent.model.named_children():
                if 'encoder' in n: enc = m; break
        
        if enc:
            for p in enc.parameters(): p.requires_grad = False
            print(colored('Encoder frozen.', 'green'))
        else:
            print(colored('Could not find encoder to freeze.', 'red'))


@hydra.main(config_name='config', config_path='.')
def train_transfer(cfg: dict):
    log_file = "terminal_log.txt"
    sys.stdout = Tee(log_file, "w")
    
    cfg.steps = max(cfg.steps, 1)
    cfg.episodic = True 
    
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)
    logger = Logger(cfg)

    checkpoint_path = cfg.get('checkpoint_path', None)
    freeze_encoder = cfg.get('freeze_encoder', False) 

    if checkpoint_path and os.path.exists(checkpoint_path):
        smart_load_checkpoint(agent, checkpoint_path, freeze_encoder)
    else:
        print(colored("WARNING: No checkpoint found! Training from scratch.", "red"))

    print(colored(f"Starting Transfer Learning on task: {cfg.task}", "green", attrs=['bold']))
    trainer = OnlineTrainer(
        cfg=cfg,
        env=env,
        agent=agent,
        buffer=buffer,
        logger=logger,
    )
    
    trainer.train()
    print('\nTransfer experiment completed.')

if __name__ == '__main__':
    train_transfer()
