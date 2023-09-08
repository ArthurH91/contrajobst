import os
import json, codecs
import matplotlib.pyplot as plt
import numpy as np

# Openning the files
path = os.getcwd()

theta_json = codecs.open(
    path + "/results/results_theta_-18_06_WS_600.json", "r", encoding="utf-8"
).read()

# Loading the json files
results = json.loads(theta_json)

theta_list = results["theta"]
q_dot = []
dist_min_obstacle = []
initial_cost = []
principal_cost = []
terminal_cost = []

obstacle_cost = []
grad = []

for theta in theta_list:
    q_dot.append(results["q_dot_" + str(round(theta, 3))])
    dist_min_obstacle.append(results["dist_min_obs_" + str(round(theta, 3))])
    initial_cost.append(results["initial_cost_" + str(round(theta, 3))])
    principal_cost.append(results["principal_cost_" + str(round(theta, 3))])
    terminal_cost.append(results["terminal_cost_" + str(round(theta, 3))])
    obstacle_cost.append(results["obstacle_cost_" + str(round(theta, 3))])
    grad.append(np.linalg.norm(results["grad_" + str(round(theta, 3))]))


###* SPEED

plt.figure()
for k in range(len(theta_list)):
    plt.plot(q_dot[k], "-o", label="theta = " + str(round(theta_list[k], 3)))

plt.xlabel("Theta")
plt.ylabel("Speed")
plt.legend()
plt.title("Speed through iterations")


###* DISTANCE MIN TO OBSTACLE
plt.figure()


for k in range(len(theta_list)):
    plt.plot(
        dist_min_obstacle[k], "-o", label="theta = " + str(round(theta_list[k], 3))
    )

plt.plot(np.zeros(len(dist_min_obstacle[k])), label="Collision")
plt.xlabel("Theta")
plt.ylabel("Distance (m)")
plt.legend()
plt.title("Distance min to obstacle through iterations")


###* EVALUATION OF COSTS

plt.figure()

plt.subplot(221)
plt.plot(theta_list, initial_cost, "o-")
plt.ylabel("Initial cost")
plt.xlabel("theta")
plt.yscale("log")
plt.title("Initial cost through theta (pose of obstacle)")

plt.subplot(222)
plt.plot(theta_list, principal_cost, "o-")
plt.ylabel("Running cost")
plt.xlabel("theta")
plt.yscale("log")
plt.title("Running cost through theta (pose of obstacle)")

plt.subplot(223)
plt.plot(theta_list, obstacle_cost, "o-")
plt.ylabel("Obstacle cost")
plt.xlabel("theta")
plt.title("Obstacle cost through theta (pose of obstacle)")

plt.subplot(224)
plt.plot(theta_list, terminal_cost, "o-")
plt.ylabel("Terminal cost")
plt.xlabel("theta")
plt.yscale("log")
plt.title("Terminal cost through theta (pose of obstacle)")
plt.suptitle("Costs through theta")

###* STUDY OF THE GRADIENT


plt.figure()
plt.plot(theta_list, grad, "-o")
plt.ylabel("Gradient")
plt.xlabel("Theta")
plt.yscale("log")
plt.title("Gradient norm through theta (pose of obstacle)")

plt.show()
