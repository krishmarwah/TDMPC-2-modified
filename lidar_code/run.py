from env import RobotEnv
import torch
import time

env = RobotEnv(gui=True)
obs = env.reset()

print("Observation keys:", obs.keys())
print("State Shape:", obs["state"].shape)

step = 0
try:
    while True:
        action = env.rand_act()
        obs, reward, done, info = env.step(action)

        if step % 10 == 0:
            print(f"step={step}, reward={reward.item():.2f}, success={info['success']}")

        if done:
            print("RESET", info)
            obs = env.reset()
            step = 0
            time.sleep(1.0)

        step += 1
        time.sleep(0.05)
except KeyboardInterrupt:
    print("Stopped.")
