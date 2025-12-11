import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer:
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.

    - Stores full episodes.
    - Uses 'episode' field as trajectory ID for SliceSampler.
    - Can store on CUDA or CPU depending on available memory.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._capacity = min(cfg.buffer_size, cfg.steps)

        # Sampler over trajectory slices
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
            strict_length=True,
            cache_values=cfg.multitask,
        )

        # Total batch size returned by ReplayBuffer.sample()
        self._batch_size = self.cfg.batch_size * (self.cfg.horizon + 1)

        # Episode counter
        self._num_eps = 0

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------
    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _reserve_buffer(self, storage):
        """Create a ReplayBuffer on top of the given storage."""
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=False,
            prefetch=0,
            batch_size=self._batch_size,
        )

    def _init(self, td: TensorDict):
        """
        Initialize the replay buffer using the first episode.

        td: TensorDict with batch size (T,) where T = episode length.
        """
        print(f"[Buffer] Initializing replay buffer")
        print(f"[Buffer] Capacity (steps): {self._capacity:,}")

        # Estimate memory per step
        mem_free = 0
        if torch.cuda.is_available():
            mem_free, _ = torch.cuda.mem_get_info()

        # td has batch size (T,); estimate bytes per step
        if isinstance(td, TensorDict):
            T = td.batch_size[0] if td.batch_size else 1
            total_bytes_episode = 0
            for v in td.values():
                if isinstance(v, TensorDict):
                    for x in v.values():
                        total_bytes_episode += x.numel() * x.element_size()
                else:
                    total_bytes_episode += v.numel() * v.element_size()
            bytes_per_step = total_bytes_episode / max(T, 1)
        else:
            raise TypeError("Expected TensorDict in _init")

        total_bytes = bytes_per_step * self._capacity
        print(f"[Buffer] Estimated storage required: {total_bytes/1e9:.2f} GB")

        # Decide device for storage
        if torch.cuda.is_available() and 2.5 * total_bytes < mem_free:
            storage_device = "cuda:0"
        else:
            storage_device = "cpu"

        print(f"[Buffer] Using {storage_device.upper()} memory for storage.")
        self._storage_device = torch.device(storage_device)

        storage = LazyTensorStorage(self._capacity, device=self._storage_device)
        return self._reserve_buffer(storage)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def load(self, td: TensorDict):
        """
        Load a batch of episodes into the buffer.

        td: TensorDict with batch size (N, T, ...) where
            N = number of episodes, T = episode length.
        """
        num_new_eps = len(td)  # N
        episode_idx = torch.arange(
            self._num_eps,
            self._num_eps + num_new_eps,
            dtype=torch.int64,
        )

        # Broadcast episode IDs over time dimension
        td["episode"] = episode_idx.unsqueeze(-1).expand(-1, td["reward"].shape[1])

        # Initialize buffer on first load
        if self._num_eps == 0:
            print("[Buffer] load(): initializing buffer from first batch")
            self._buffer = self._init(td[0])

        # Flatten (N, T, ...) -> (N*T, ...)
        td_flat = td.reshape(td.shape[0] * td.shape[1])
        self._buffer.extend(td_flat)
        self._num_eps += num_new_eps

        print(f"[Buffer] load(): now has {self._num_eps} episodes")
        return self._num_eps

    def add(self, td: TensorDict):
        """
        Add a single episode to the buffer.

        td: TensorDict with batch size (T,) for one episode.
        """
        # Tag this episode with its ID
        td["episode"] = torch.full_like(
            td["reward"],
            self._num_eps,
            dtype=torch.int64,
        )

        # Initialize underlying replay buffer on first episode
        if self._num_eps == 0:
            print("[Buffer] add(): first episode, initializing replay buffer")
            self._buffer = self._init(td)

        print(f"[Buffer] add(): adding episode #{self._num_eps}")
        print(f"[Buffer]         td.batch_size = {td.batch_size}")
        print(f"[Buffer]         keys = {list(td.keys())}")

        self._buffer.extend(td)
        self._num_eps += 1

        print(f"[Buffer] add(): total episodes in buffer = {self._num_eps}")
        return self._num_eps

    # ---------------------------------------------------------------------
    # Sampling for training
    # ---------------------------------------------------------------------
    def _prepare_batch(self, td: TensorDict):
        """
        Prepare a sampled batch for TD-MPC2.

        Input:
            td: TensorDict with batch size (T, B)
                T = horizon+1, B = batch size

        Output:
            obs:        (T, B, obs_dim)
            action:     (T-1, B, act_dim)
            reward:     (T-1, B, 1)
            terminated: (T-1, B, 1)
            task:       (B,) or None
        """
        td = td.select("obs", "action", "reward", "terminated", "task", strict=False)
        td = td.to(self._device, non_blocking=True)

        obs = td.get("obs").contiguous()
        action = td.get("action")[1:].contiguous()
        reward = td.get("reward")[1:].unsqueeze(-1).contiguous()

        terminated = td.get("terminated", None)
        if terminated is not None:
            terminated = terminated[1:].unsqueeze(-1).contiguous()
        else:
            terminated = torch.zeros_like(reward)

        task = td.get("task", None)
        if task is not None:
            task = task[0].contiguous()

        return obs, action, reward, terminated, task

    def sample(self):
        """
        Sample a batch of subsequences from the buffer.

        Returns:
            obs, action, reward, terminated, task
        """
        if not hasattr(self, "_buffer") or self._num_eps == 0:
            raise RuntimeError(
                "ReplayBuffer not initialized yet. "
                "Add at least one episode before sampling."
            )

        # ReplayBuffer.sample() returns a flat batch of size (batch_size,)
        td = self._buffer.sample()

        # Reshape to (T, B): T = horizon+1, B = batch_size
        td = td.view(-1, self.cfg.horizon + 1).permute(1, 0)

        return self._prepare_batch(td)
