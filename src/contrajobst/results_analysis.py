import json, codecs
import matplotlib.pyplot as plt
import numpy as np

# Openning the files
conv_json = codecs.open("results_conv.json", "r", encoding="utf-8").read()
it_76_json = codecs.open("results_it_76.json", "r", encoding="utf-8").read()
it_77_json = codecs.open("results_it_77.json", "r", encoding="utf-8").read()
it_78_json = codecs.open("results_it_78.json", "r", encoding="utf-8").read()
it_79_json = codecs.open("results_it_79.json", "r", encoding="utf-8").read()
it_80_json = codecs.open("results_it_80.json", "r", encoding="utf-8").read()

# Loading the json files
conv = json.loads(conv_json)
it_76 = json.loads(it_76_json)
it_77 = json.loads(it_77_json)
it_78 = json.loads(it_78_json)
it_79 = json.loads(it_79_json)
it_80 = json.loads(it_80_json)


###* Q_TRS
Q_trs_conv = conv["Q_trs"]
Q_trs_76 = it_76["Q_trs"]
Q_trs_77 = it_77["Q_trs"]
Q_trs_78 = it_78["Q_trs"]
Q_trs_79 = it_79["Q_trs"]
Q_trs_80 = it_80["Q_trs"]


###* SPEED

q_dot_conv = conv["q_dot"]
q_dot_76 = it_76["q_dot"]
q_dot_77 = it_77["q_dot"]
q_dot_78 = it_78["q_dot"]
q_dot_79 = it_79["q_dot"]
q_dot_80 = it_80["q_dot"]

plt.subplot(211)
plt.plot(q_dot_76, "o-", label=" Iteration 76")
plt.plot(q_dot_77, "o-", label=" Iteration 77")
plt.plot(q_dot_78, "o-", label=" Iteration 78")
plt.plot(q_dot_79, "o-", label=" Iteration 79")
plt.plot(q_dot_80, "o-", label=" Iteration 80")
plt.plot(q_dot_conv, "o-", label=" Iteration conv (96)")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Speed")
plt.title("Evolution of speed through iterations")

###* DISTANCE MIN TO OBSTACLE

dist_min_obs_conv = conv["dist_min_obs"]
dist_min_obs_76 = it_76["dist_min_obs"]
dist_min_obs_77 = it_77["dist_min_obs"]
dist_min_obs_78 = it_78["dist_min_obs"]
dist_min_obs_79 = it_79["dist_min_obs"]
dist_min_obs_80 = it_80["dist_min_obs"]

plt.subplot(212)
plt.plot(dist_min_obs_76, "-o", label=" Iteration 76")
plt.plot(dist_min_obs_77, "-o", label=" Iteration 77")
plt.plot(dist_min_obs_78, "-o", label=" Iteration 78")
plt.plot(dist_min_obs_79, "-o", label=" Iteration 79")
plt.plot(dist_min_obs_80, "-o", label=" Iteration 80")
plt.plot(dist_min_obs_conv, "-o", label=" Iteration conv (96)")
plt.plot(np.zeros(len(dist_min_obs_conv)), label="Collision")
plt.ylabel("Distance (m)")
plt.xlabel("Iterations")
plt.legend()
plt.title("Distance min of robot to obstacle")

plt.suptitle("Evaluation of distance to obstacle and speed of the robot")

###* EVALUATION OF COSTS

# INITIAL COST

initial_cost = [
    it_76["initial_cost"],
    it_77["initial_cost"],
    it_78["initial_cost"],
    it_79["initial_cost"],
    it_80["initial_cost"],
    conv["initial_cost"],
]

plt.figure()
plt.subplot(221)

plt.plot(initial_cost, "o-")
plt.ylabel("Initial cost")
plt.xlabel("Iterations (76 to conv)")
plt.yscale("log")
plt.title("Initial cost through iterations")

# PRINCIPAL COST

principal_cost = [
    it_76["principal_cost"],
    it_77["principal_cost"],
    it_78["principal_cost"],
    it_79["principal_cost"],
    it_80["principal_cost"],
    conv["principal_cost"],
]

plt.subplot(222)

plt.plot(principal_cost, "o-")
plt.ylabel("principal cost")
plt.xlabel("Iterations (76 to conv)")
plt.yscale("log")
plt.title("principal cost through iterations")

# OBSTACLE COST

obstacle_cost = [
    it_76["obstacle_cost"],
    it_77["obstacle_cost"],
    it_78["obstacle_cost"],
    it_79["obstacle_cost"],
    it_80["obstacle_cost"],
    conv["obstacle_cost"],
]

plt.subplot(223)

plt.plot(obstacle_cost, "o-")
plt.ylabel("obstacle cost")
plt.xlabel("Iterations (76 to conv)")
plt.yscale("log")
plt.title("obstacle cost through iterations")


# TERMINAL COST

terminal_cost = [
    it_76["terminal_cost"],
    it_77["terminal_cost"],
    it_78["terminal_cost"],
    it_79["terminal_cost"],
    it_80["terminal_cost"],
    conv["terminal_cost"],
]

plt.subplot(224)

plt.plot(terminal_cost, "o-")
plt.ylabel("terminal cost")
plt.xlabel("Iterations (76 to conv)")
plt.yscale("log")
plt.title("terminal cost through iterations")
plt.suptitle("Study of the costs through iterations")


### STUDY OF THE GRADIENT


grad_conv = conv["grad"]
grad_76 = it_76["grad"]
grad_77 = it_77["grad"]
grad_78 = it_78["grad"]
grad_79 = it_79["grad"]
grad_80 = it_80["grad"]

# Norm of the gradient
norm_grad = [
    np.linalg.norm(grad_76),
    np.linalg.norm(grad_77),
    np.linalg.norm(grad_78),
    np.linalg.norm(grad_79),
    np.linalg.norm(grad_80),
    np.linalg.norm(grad_conv),
]

plt.figure()
plt.plot(norm_grad)
plt.ylabel("Gradient")
plt.xlabel("Iterations")
plt.yscale("log")
plt.title("Gradient norm through iterations")
plt.show()
