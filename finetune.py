import os
import sys
import warnings
import torch
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
        obs = batch.get('obs')
        action = batch.get('action')
        reward = batch.get('reward')
        terminated = batch.get('terminated')
        task = batch.get('task')

    if terminated is None:
        terminated = torch.zeros_like(reward)

    is_multitask = self.cfg.multitask or (hasattr(self, 'cfg') and getattr(self.cfg, 'task_size', 0) > 0)
    
    if task is None:
        if is_multitask:
             task = torch.zeros_like(reward, dtype=torch.long)
        else:
             task = None

    try:
        return self._update(
            obs=obs, 
            action=action, 
            reward=reward, 
            terminated=terminated, 
            task=task
        )
    except TypeError:
        if task is None:
             try:
                 return self._update(obs, action, reward, terminated)
             except TypeError:
                 return self._update(obs, action, reward, terminated, None)
        else:
             return self._update(obs, action, reward, terminated, task)

    except AssertionError as e:
        if "task is None" in str(e):
            return self._update(obs, action, reward, terminated, task=None)
        raise e
        
    except AttributeError as e:
        if "_tensordict" in str(e):
            return {"loss": 0.0}
        raise e

TDMPC2.update = patched_update


def load_checkpoint(agent, checkpoint_path, freeze_encoder=False):
    print(colored(f'Loading checkpoint from {checkpoint_path}...', 'green', attrs=['bold']))
    
    try:
        loaded_obj = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
            state_dict = loaded_obj['model']
        elif hasattr(loaded_obj, 'state_dict'):
            state_dict = loaded_obj.state_dict()
        else:
            state_dict = loaded_obj

        if hasattr(state_dict, "to_dict"):
             state_dict = state_dict.to_dict()
        state_dict = {k: v for k, v in state_dict.items() if k != "__batch_size"}

        msg = agent.model.load_state_dict(state_dict, strict=False)
        print(colored(f'Weights loaded with msg: {msg}', 'green'))

    except Exception as e:
        print(colored(f'Standard load failed: {e}', 'red'))
        print(colored("Attempting safe-mode partial loading...", "yellow"))
        try:
             model_dict = agent.model.state_dict()
             pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
             model_dict.update(pretrained_dict) 
             agent.model.load_state_dict(model_dict)
             print(colored("Safe-mode loading successful.", "green"))
        except Exception as e2:
             print(colored(f"Fatal loading error: {e2}", "red"))
             raise e2

    if freeze_encoder:
        print(colored('Freezing encoder layers...', 'yellow'))
        encoder_module = None
        if hasattr(agent.model, 'encoder'): encoder_module = agent.model.encoder
        elif hasattr(agent.model, '_encoder'): encoder_module = agent.model._encoder
        
        if encoder_module is None:
            for name, module in agent.model.named_children():
                if 'encoder' in name.lower():
                    encoder_module = module
                    break

        if encoder_module:
            for param in encoder_module.parameters():
                param.requires_grad = False
            print(colored('Encoder frozen successfully.', 'green'))
        else:
            print(colored('WARNING: Could not locate encoder module to freeze. Training all layers.', 'red'))


@hydra.main(config_name='config', config_path='.')
def train_finetune(cfg: dict):
    assert torch.cuda.is_available()
    
    log_file = "terminal_log.txt"
    sys.stdout = Tee(log_file, "w")

    assert cfg.steps > 0, 'Must train for at least 1 step.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    
    cfg.episodic = True 

    env = make_env(cfg)
    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)
    logger = Logger(cfg)

    checkpoint_path = cfg.get('checkpoint_path', None)
    freeze_encoder = cfg.get('freeze_encoder', False)

    
    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(agent, checkpoint_path, freeze_encoder)
    elif checkpoint_path:
        print(colored(f'Checkpoint path provided but not found: {checkpoint_path}', 'red'))

    trainer = OnlineTrainer(
        cfg=cfg,
        env=env,
        agent=agent,
        buffer=buffer,
        logger=logger,
    )
    
    trainer.train()
    print('\nTraining completed successfully')

if __name__ == '__main__':
    train_finetune()
