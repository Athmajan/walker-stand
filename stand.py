import matplotlib.pyplot as plt
from dm_control import suite
import numpy as np

# Create the HalfCheetah environment
env = suite.load(domain_name="walker", task_name="stand")

# Reset the environment to get the initial state
state = env.reset()


def visualize(env):
    frameA = np.hstack([env.physics.render(480, 480, camera_id=0),
                        env.physics.render(480, 480, camera_id=1)])
    plt.imshow(frameA)
    plt.pause(0.01)  # Need min display time > 0.0.
    plt.axis('off') 
    plt.draw()      
    plt.close() 
    return





action_spec = env.action_spec()
time_step = env.reset()
i = 0
while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                            action_spec.maximum,
                            size=action_spec.shape)
    
    time_step = env.step(action)
    visualize(env)
    
    

    
# Close the environment
env.close()
