import numpy
import glob
import sys
import matplotlib.pyplot as plt
import os
import time

from pawn import Agent
from environment import Environment

def mate_and_mutate(world_size, best_models):
    """Pick two parents and mate them, then mutate the child"""

    agent = Agent(world_size)
    starting_from = numpy.random.choice(best_models)
    print(f"Starting from {starting_from}")
    agent.brain.load(starting_from)

    agent.brain.copy_weights_and_mutate_point(None)
    agent.memory = numpy.array([0] * agent.brain.n_memory)

    return agent


def evolve_brains(seed):

    # set the random seed
    numpy.random.seed(seed)

    chunk_size = 300
    vision_distance = chunk_size
    n_chunks = 3
    world_size = n_chunks * chunk_size

    n_starting_agents = 100
    n_test_per_agent = 20
    current_best_score = 0

    while True:

        print()

        previous_models = glob.glob(f"models/*__*.pkl")

        if len(previous_models) > n_starting_agents:
            best_models = sorted(
                glob.glob(f"models/*__*.pkl"),
                key=lambda x: int(x.split("__")[-1].split(".")[0])
            )[::-1][:10]
            current_best_score = int(best_models[0].split("__")[-1].split(".")[0])
            agent = mate_and_mutate(world_size, best_models)
        else:
            agent = Agent(world_size)

        print(f"Agent number: {len(previous_models) + 1}, named: {agent.name}:")
        print(agent.brain.n_hidden, agent.brain.n_memory)

        scores = []
        for e in range(n_test_per_agent):
            agent.reset()
            env = Environment(world_size, vision_distance)
            #t1 = time.time()
            score = env.test_agent(agent, display=False)
            scores.append(score)
            #print(f"Epoch {e}, score {score}")
            #print(f"FPS: {score / (time.time() - t1)}")
            if score < current_best_score / 10:
                print(f"Score {score} too low (< {current_best_score / 10}), skipping")
                break
        else:
            mean_score = numpy.mean(scores)
            print(f"Mean score: {mean_score}")
            agent.brain.save(agent.name, round(mean_score))

        del agent

def plot_scores():

    def smooth(y, box_pts):
        box = numpy.ones(box_pts) / box_pts
        y_smooth = numpy.convolve(y, box, mode='same')
        return y_smooth

    for dir in ["models1", "models2", "models"]:

        scores = []
        max_ = []

        files = list(glob.glob(f"{dir}/*__*.pkl"))
        files.sort(key=lambda x: os.path.getmtime(x))

        for fp in files:
            split = fp.split("__")
            scores.append(int(split[-1].split(".")[0]))

        for sc in scores:

            if not max_:
                max_.append(sc)
            else:
                lmax = sc if sc > max_[-1] else max_[-1]
                max_.append(lmax)

        smoothing_window = 100
        indexes = list(range(len(scores)))
        #plt.plot(indexes, scores)
        #plt.plot(indexes[:-smoothing_window], smooth(scores, smoothing_window)[:-smoothing_window])
        plt.plot(indexes, max_)

    plt.xlabel("Evaluations")
    plt.ylabel("Score")
    plt.ylim(0, numpy.max(scores) + 1000)

    plt.show()

def check_no_food(path):

    chunk_size = 300
    vision_distance = chunk_size
    n_chunks = 3
    world_size = n_chunks * chunk_size

    agent = Agent(world_size, position=None, from_file=path)
    agent.reset()

    env = Environment(world_size, vision_distance, n_food=10)
    env.test_agent(agent)


def check_score_memory(path="models/14__OOaknwVUiZvyIQ__601.pkl", display=True):

    chunk_size = 300
    vision_distance = chunk_size
    n_chunks = 3
    world_size = n_chunks * chunk_size

    memoriess = []
    positionss = []
    directionss = []

    for i in range(20):

        agent = Agent(world_size)
        agent.brain.load(path)
        agent.reset()
        env = Environment(world_size, vision_distance)
        score, memories, positions, directions = env.test_agent_memory(agent, True)

        memoriess += memories
        positionss += positions
        directionss += directions

        del agent

    # For each memory
    for i in range(20):
        for e, ms in enumerate(memoriess):
            if ms[i] > 0.5:
                plt.scatter(positionss[e][0], positionss[e][1], c="C0")
        plt.show()
        plt.clf()

def check_score(path="models/14__OOaknwVUiZvyIQ__601.pkl"):

    chunk_size = 300
    vision_distance = chunk_size
    n_chunks = 3
    world_size = n_chunks * chunk_size

    scores = []
    for i in range(100):

        agent = Agent(world_size)
        agent.brain.load(path)
        agent.reset()
        env = Environment(world_size, vision_distance)
        scores.append(env.test_agent(agent, False))
        print(scores[-1])
        del agent

    print(path)
    print("Scores: ", scores)
    print("Mean score: ", numpy.mean(scores))

def check_score_ablation(path):

    chunk_size = 300
    vision_distance = chunk_size
    n_chunks = 3
    world_size = n_chunks * chunk_size

    scores = []
    scores_ablated = []
    scores_ablated_hunger = []
    scores_ablated_both = []

    for i in range(50):
        agent = Agent(world_size)
        agent.brain.load(path)
        agent.reset()
        env = Environment(world_size, vision_distance)
        scores.append(env.test_agent(agent, display=False, no_memory=False))
        print(i, scores[-1])
        del agent

    for i in range(50):
        agent = Agent(world_size)
        agent.brain.load(path)
        agent.reset()
        env = Environment(world_size, vision_distance)
        scores_ablated.append(env.test_agent(agent, display=False, no_memory=True))
        print(i, scores_ablated[-1])
        del agent

    for i in range(50):
        agent = Agent(world_size)
        agent.brain.load(path)
        agent.reset()
        env = Environment(world_size, vision_distance)
        scores_ablated_hunger.append(env.test_agent(agent, display=False, no_memory=False, no_hunger_signal=True))
        print(i, scores_ablated_hunger[-1])
        del agent

    for i in range(50):
        agent = Agent(world_size)
        agent.brain.load(path)
        agent.reset()
        env = Environment(world_size, vision_distance)
        scores_ablated_both.append(env.test_agent(agent, display=False, no_memory=True, no_hunger_signal=True))
        print(i, scores_ablated_both[-1])
        del agent

    print(path)
    print("Scores: ", scores)
    print("Mean score: ", numpy.mean(scores))
    print("Scores ablated: ", scores_ablated)
    print("Mean score ablated: ", numpy.mean(scores_ablated))
    print("Scores ablated signals: ", scores_ablated_hunger)
    print("Mean score ablated signals: ", numpy.mean(scores_ablated_hunger))
    print("Scores ablated both: ", scores_ablated_both)
    print("Mean score ablated both: ", numpy.mean(scores_ablated_both))


if __name__ == "__main__":

    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 42

    # /usr/local/opt/python@3.10/bin/python3

    #check_score_ablation("models/9gWCKLp73rE9_A__4248.pkl")
    #check_no_food(path=brain_path)
    #check_place_grid_cell(path=brain_path)
    #plot_scores()
    evolve_brains(seed=seed)
    #check_score_memory("models/6q6nNrUFVzV9lw__4396.pkl", True)
    #check_score_memory("models/9gWCKLp73rE9_A__4248.pkl")
