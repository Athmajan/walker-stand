import dmc2gym

env = dmc2gym.make(domain_name='walker', task_name='stand', seed=1)

done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)