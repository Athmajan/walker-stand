import matplotlib.pyplot as plt
from dm_control import suite
import numpy as np
import cv2

# Create the Walker environment
env = suite.load(domain_name="walker", task_name="stand")

# Function to visualize the environment
def visualize(env):
    frameA = np.hstack([env.physics.render(480, 480, camera_id=0),
                        env.physics.render(480, 480, camera_id=1)])
    return frameA

# Create a list to store the frames
frames = []

action_spec = env.action_spec()
time_step = env.reset()
while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                               action_spec.maximum,
                               size=action_spec.shape)
    time_step = env.step(action)
    frame = visualize(env)
    frames.append(frame)

# Close the environment
env.close()

# Define the codec and create a VideoWriter object
height, width, _ = frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('walker_stand_animation.mp4', fourcc, 20.0, (width, height))

# Write the frames to the video file
for frame in frames:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)

# Release the video writer
out.release()

# Display the video using OpenCV
cap = cv2.VideoCapture('walker_stand_animation.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Walker Stand Animation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
