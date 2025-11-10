"""
GRPO Skeleton: Minimal Asynchronous Training Loop
------------------------------------------------
Students should complete the TODO parts to:
 - implement rollout generation (Generator)
 - compute rewards (Scorer)
 - perform policy updates (Learner)
 - synchronize model weights between Generator and Learner
"""

import asyncio
import argparse
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ===================== Basic setup =====================

G = 4  # group size (number of actions per prompt)
STATE_DIM = 4
ACTION_DIM = 4


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Data container =====================

class Trajectory:
    """A single rollout sample."""
    def __init__(self, version, state, actions, logps, rewards):
        self.version = version
        self.state = state
        self.actions = actions
        self.logps = logps
        self.rewards = rewards


# ===================== Actors =====================

@ray.remote
class TrajectoryQueue:
    """Buffer between Generator and Scorer."""
    def __init__(self):
        self.q = asyncio.Queue()

    async def put(self, traj: Trajectory):
        await self.q.put(traj)

    async def get(self):
        try:
            return await asyncio.wait_for(self.q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            return None


@ray.remote
class ReplayBuffer:
    """Stores scored trajectories for the Learner."""
    def __init__(self):
        self.data = []

    async def put(self, traj: Trajectory):
        # TODO: store completed trajectories here
        pass

    async def sample(self, k: int):
        # TODO: sample k trajectories for training
        return []


@ray.remote
class Scorer:
    """Assigns rewards to generated actions."""
    def __init__(self, traj_q, replay_buf):
        self.traj_q = traj_q
        self.replay_buf = replay_buf
        self.running = False

    async def run(self):
        """Continuously fetch trajectories, assign rewards, and store them."""
        self.running = True
        while self.running:
            traj = await self.traj_q.get.remote()
            if traj is None:
                continue
            # TODO: compute rewards for traj.actions
            await self.replay_buf.put.remote(traj)

    async def stop(self):
        self.running = False


@ray.remote
class Learner:
    """Learns policy updates from the replay buffer."""
    def __init__(self, replay_buf):
        self.device = get_device()
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, ACTION_DIM),
        ).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)
        self.version = 0
        self.replay_buf = replay_buf

    async def step(self):
        """One GRPO/PPO-style update step."""
        # TODO: sample from replay buffer, compute advantages, update model
        loss = torch.tensor(0.0)
        self.version += 1
        return float(loss.item())

    async def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    async def get_version(self):
        return self.version


@ray.remote
class Generator:
    """Generates rollouts using the current policy."""
    def __init__(self, traj_q):
        self.device = get_device()
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, ACTION_DIM),
        ).to(self.device)
        self.traj_q = traj_q
        self.version = 0

    async def generate(self, state: torch.Tensor):
        """Generate actions and send to Scorer."""
        # TODO: sample actions and log-probs
        pass

    async def update(self, weights, version: int):
        """Load updated learner weights."""
        sd = self.model.state_dict()
        for n, w in weights.items():
            sd[n] = w.to(self.device)
        self.model.load_state_dict(sd)
        self.version = version


# ===================== Training loop =====================

def run_once(num_steps: int = 3):
    traj_q = TrajectoryQueue.remote()
    replay_buf = ReplayBuffer.remote()
    learner = Learner.remote(replay_buf)
    scorer = Scorer.remote(traj_q, replay_buf)
    generator = Generator.remote(traj_q)

    # TODO: Driver code for the training loop
    pass



# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    run_once(num_steps=args.steps)
