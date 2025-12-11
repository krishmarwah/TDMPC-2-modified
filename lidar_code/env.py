import os
import math
import time
import numpy as np
import torch
import pybullet as p
import pybullet_data
from gym import spaces


class RobotEnv:
    def __init__(self, gui=False, seed=None):
        self.gui = gui
        if p.isConnected():
            self.client = p.getConnectionInfo()["userConnectionIdentifier"]
        else:
            self.client = p.connect(p.GUI if gui else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # ENV PARAMETERS
        self.num_rays = 120
        self.max_dist = 5.0  # meters
        self.arena_limit = 4.5  # world bound
        self.max_speed = 0.6  # m/s (linear)
        self.max_turn = 1.5  # rad/s (angular)
        self.goal_radius = 0.35  # success if within this distance (meters)
        self.collision_threshold = 0.05  # hit fraction threshold (lower fraction = closer object); increased to be robust
        self.step_penalty = -0.05
        self.collision_penalty = -20.0
        self.success_reward = 40.0
        self.out_of_bounds_penalty = -20.0
        self.progress_scale = 15.0  # multiplies change in distance-to-goal
        self.dt = (
            1.0 / 10.0
        )  # control timestep: 0.1s. Keep consistent with training assumptions.

        # start/goal (can randomize later)
        self.start = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.goal = torch.tensor([3.0, 3.0], dtype=torch.float32)
        # self.goal = torch.tensor([1.5, 1.5], dtype=torch.float32)

        # robot z height for rays and rendering
        self.robot_z = 0.1
        self.ray_start_z = self.robot_z + 0.05

        self.step_count = 0
        self.max_steps = 300

        self._rng = np.random.RandomState(seed if seed is not None else 0)
        self.seed(seed)

        # Build action/observation spaces
        self._action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # lidar 120 normalized (0..1) + dist_to_goal (meters) + angle_to_goal (radians)
        self._obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(122,), dtype=np.float32
        )

        # Initialize / reset simulation
        self._last_pos = None
        self._last_dist = None
        self.reset()

    # gym-like properties
    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._obs_space

    # seeding
    def seed(self, seed=None):
        if seed is None:
            seed = int(time.time() % 1e6)
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        self._rng.seed(self._seed)

    # reset
    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)

        self.step_count = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Load flat plane and (optionally) walls/obstacles
        p.loadURDF("plane.urdf")
        self._create_walls()
        self._spawn_fixed_obstacles()

        # Load robot visual/URDF if present, otherwise use a small box for display
        base_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(base_dir, "bot.urdf")
        if os.path.exists(urdf_path):
            self.robot = p.loadURDF(
                urdf_path, [self.start[0].item(), self.start[1].item(), self.robot_z]
            )
        else:
            # small cube visual body
            self.robot = p.loadURDF(
                "r2d2.urdf", [self.start[0].item(), self.start[1].item(), self.robot_z]
            )

        # visualization of goal
        try:
            self.goal_vis = p.loadURDF(
                "sphere_small.urdf", [self.goal[0].item(), self.goal[1].item(), 0.08]
            )
        except Exception:
            self.goal_vis = p.loadURDF(
                "cube.urdf",
                [self.goal[0].item(), self.goal[1].item(), 0.08],
                globalScaling=0.15,
            )

        # Pose state (x, y, yaw) stored separately for kinematic integration
        self._pose = np.array(
            [self.start[0].item(), self.start[1].item(), 0.0], dtype=np.float32
        )
        self._apply_pose_to_pybullet()

        self.prev_dist = self._goal_distance()
        self._last_dist = self.prev_dist.clone()
        self._last_pos = torch.tensor(self._pose[:2]).clone()

        # Return initial observation dict
        return self._get_obs()

    # main step
    def step(self, action):
        self.step_count += 1

        # Convert action to numpy
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy().flatten()
        else:
            action = np.asarray(action).flatten()

        action = np.clip(action, -1.0, 1.0)
        v_cmd = float(action[0]) * self.max_speed
        w_cmd = float(action[1]) * self.max_turn

        # Kinematic integration (unicycle model)
        x, y, yaw = self._pose
        # forward velocities in world frame
        dx = v_cmd * math.cos(yaw) * self.dt
        dy = v_cmd * math.sin(yaw) * self.dt
        dyaw = w_cmd * self.dt

        # update pose
        x_new = x + dx
        y_new = y + dy
        yaw_new = ((yaw + dyaw) + math.pi) % (2.0 * math.pi) - math.pi

        self._pose = np.array([x_new, y_new, yaw_new], dtype=np.float32)
        self._apply_pose_to_pybullet()

        # step physics for visual consistency only
        p.stepSimulation()
        if self.gui:
            time.sleep(self.dt)

        obs = self._get_obs()  # dict -> {"state": tensor(122,)}

        # compute reward
        lidar = obs["state"][:-2]  # normalized hit fractions
        dist_goal = obs["state"][-2]  # scalar tensor
        # angle = obs['state'][-1]      # not used in shaping, but available

        reward = 0.0
        done = False
        info = {"success": False}

        # success
        if dist_goal.item() < self.goal_radius:
            reward = self.success_reward
            info["success"] = True
            done = True
        # collision (very close obstacle)
        elif torch.min(lidar).item() < self.collision_threshold:
            reward = self.collision_penalty
            done = True
        # out of bounds
        elif (
            abs(self._pose[0]) > self.arena_limit
            or abs(self._pose[1]) > self.arena_limit
        ):
            reward = self.out_of_bounds_penalty
            done = True
        else:
            # shaping reward: progress towards goal
            dist_now = dist_goal
            delta = self.prev_dist - dist_now
            self.prev_dist = dist_now
            reward += float(self.progress_scale * delta.item())
            reward += self.step_penalty

        if self.step_count >= self.max_steps:
            done = True

        return obs, torch.tensor(reward, dtype=torch.float32), done, info

    # observation construction
    def _get_obs(self):
        lidar = (
            self._get_lidar()
        )  # shape (num_rays,) as torch tensor normalized 0..1 (1.0 = max_dist no hit)
        dist_to_goal = self._goal_distance()  # torch scalar (meters)
        angle_to_goal = self._goal_angle()  # torch scalar (radians)

        state = torch.cat(
            [lidar, dist_to_goal.view(1), angle_to_goal.view(1)]
        ).float()  # shape (122,)
        return {"state": state}

    # distance and angle helpers
    def _goal_distance(self):
        pos = torch.tensor(self._pose[:2], dtype=torch.float32)
        return torch.norm(pos - self.goal)

    def _goal_angle(self):
        x, y, yaw = self._pose
        robot_pos = torch.tensor([x, y], dtype=torch.float32)
        vec = self.goal - robot_pos
        goal_yaw = math.atan2(float(vec[1].item()), float(vec[0].item()))
        angle = goal_yaw - yaw
        # wrap into [-pi, pi]
        angle = ((angle + math.pi) % (2 * math.pi)) - math.pi
        return torch.tensor(angle, dtype=torch.float32)

    # lidar implementation
    def _get_lidar(self):
        # get robot base position for ray start
        pos = self._pose  # numpy array [x,y,yaw]
        x, y, yaw = pos[0], pos[1], pos[2]

        starts = []
        ends = []
        for i in range(self.num_rays):
            angle = -math.pi + 2.0 * math.pi * i / self.num_rays
            sx = x
            sy = y
            sz = self.ray_start_z
            ex = x + self.max_dist * math.cos(angle)
            ey = y + self.max_dist * math.sin(angle)
            ez = self.ray_start_z
            starts.append([sx, sy, sz])
            ends.append([ex, ey, ez])

        results = p.rayTestBatch(
            starts, ends
        )  # list of (hitObjectUid, linkIndex, hitFraction, hitPos, hitNormal)
        # extract hit fractions: if nothing hit, hitFraction == 1.0 typically
        fractions = [res[2] for res in results]
        fractions = np.array(fractions, dtype=np.float32)
        # clamp/ensure numeric stability
        fractions = np.clip(fractions, 0.0, 1.0)
        # convert into torch tensor and return
        return torch.from_numpy(fractions)  # shape (num_rays,)

    # support utilities
    def _apply_pose_to_pybullet(self):
        x, y, yaw = float(self._pose[0]), float(self._pose[1]), float(self._pose[2])
        pos = [x, y, self.robot_z]
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        try:
            p.resetBasePositionAndOrientation(self.robot, pos, orn)
        except Exception:
            # in case reset fails, try reconnecting client
            pass
        # also move goal visual if exists
        try:
            if hasattr(self, "goal_vis"):
                p.resetBasePositionAndOrientation(
                    self.goal_vis,
                    [self.goal[0].item(), self.goal[1].item(), 0.08],
                    p.getQuaternionFromEuler([0, 0, 0]),
                )
        except Exception:
            pass

    def _create_walls(self):
        scale = 5
        try:
            p.loadURDF("cube.urdf", [5, 0, 0.5], globalScaling=scale)
            p.loadURDF("cube.urdf", [-5, 0, 0.5], globalScaling=scale)
            p.loadURDF("cube.urdf", [0, 5, 0.5], globalScaling=scale)
            p.loadURDF("cube.urdf", [0, -5, 0.5], globalScaling=scale)
        except Exception:
            pass

    def _spawn_fixed_obstacles(self):
        coords = [(1.0, 1.0), (2.0, -1.0), (-1.5, 1.8), (-2.0, -1.0)]
        try:
            for x, y in coords:
                p.loadURDF("cube.urdf", [x, y, 0.2], globalScaling=0.5)
        except Exception:
            pass

    def rand_act(self):
        """Return torch tensor action in [-1, 1]"""
        return torch.from_numpy(self._rng.uniform(-1.0, 1.0, size=(2,))).float()

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    env = RobotEnv(gui=True, seed=0)
    obs = env.reset()
    print("Obs keys:", obs.keys())
    print("State shape:", obs["state"].shape)
    done = False
    steps = 0
    try:
        while not done and steps < 200:
            a = env.rand_act()
            obs, reward, done, info = env.step(a)
            if steps % 10 == 0:
                print(
                    f"step={steps}, reward={reward.item():.3f}, min_lidar={float(obs['state'][:-2].min()):.3f}, dist={obs['state'][-2].item():.3f}"
                )
            steps += 1
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()
