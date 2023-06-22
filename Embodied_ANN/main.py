import numpy
import glob
import sys
import matplotlib.pyplot as plt
import os

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
    n_test_per_agent = 10
    current_best_score = 0

    while True:

        print()

        previous_models = glob.glob(f"models/*__*.npy")

        if len(previous_models) > n_starting_agents:
            best_models = sorted(
                glob.glob(f"models/*__*.npy"),
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
            score = env.test_agent(agent, False)
            scores.append(score)
            print(f"Epoch {e}, score {score}")
            if score < current_best_score / 5:
                print(f"Score {score} too low (< {current_best_score / 5}), skipping")
                break

        mean_score = numpy.mean(scores)
        print(f"Mean score: {mean_score}")

        agent.brain.save(agent.name, round(mean_score))

        del agent


def plot_scores():

    def smooth(y, box_pts):
        box = numpy.ones(box_pts) / box_pts
        y_smooth = numpy.convolve(y, box, mode='same')
        return y_smooth

    for dir in ["models_OLD", "models_OLD2", "models"]:

        scores = []
        max_ = []

        files = list(glob.glob(f"{dir}/*__*.npy"))
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
        plt.plot(indexes[:-smoothing_window], smooth(scores, smoothing_window)[:-smoothing_window])
        plt.plot(indexes, max_)

    plt.xlabel("Evaluations")
    plt.ylabel("Score")
    plt.ylim(0, numpy.max(scores) + 1000)

    plt.show()


def check_place_grid_cell(path):

    chunk_size = 300
    vision_distance = chunk_size
    n_chunks = 3
    world_size = n_chunks * chunk_size

    agent = Agent(world_size, position=None, from_file=path)
    agent.reset()

    env = Environment(world_size, vision_distance)
    position, spike_trains = env.test_grid_cells(agent)

    neuron_ids = []
    for spike_train in spike_trains:
        if spike_train is not None:
            neuron_ids += list(spike_train['senders'])
    neuron_ids = set(neuron_ids)

    import matplotlib.pyplot as plt

    for neuron_id in neuron_ids:
        canvas = numpy.zeros((world_size, world_size))
        for i, (spike_train, pos) in enumerate(zip(spike_trains, position)):
            if spike_train is not None:
                canvas[int(pos[0]), int(pos[1])] += list(spike_train["senders"]).count(neuron_id)

        plt.imshow(canvas)
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


def check_score(path="models/14__OOaknwVUiZvyIQ__601.npy"):

    chunk_size = 300
    vision_distance = chunk_size
    n_chunks = 3
    world_size = n_chunks * chunk_size

    scores = []
    for i in range(10):

        agent = Agent(world_size)
        agent.brain.load(path)
        agent.reset()
        env = Environment(world_size, vision_distance)
        scores.append(env.test_agent(agent, True))
        del agent

    print(path)
    print("Scores: ", scores)
    print("Mean score: ", numpy.mean(scores))


def check_score_ablation_recurrent(path):

    chunk_size = 300
    vision_distance = chunk_size
    n_chunks = 3
    world_size = n_chunks * chunk_size

    scores = []
    scores_ablated = []

    for i in range(10):

        agent = Agent(world_size)
        agent.brain.load(path)
        agent.reset()
        env = Environment(world_size, vision_distance)
        scores.append(env.test_agent(agent, display=True, no_memory=False))
        del agent

    for i in range(10):

        agent = Agent(world_size)
        agent.brain.load(path)
        agent.reset()
        env = Environment(world_size, vision_distance)
        scores_ablated.append(env.test_agent(agent, display=False, no_memory=True))
        del agent

    print(path)
    print("Scores: ", scores)
    print("Mean score: ", numpy.mean(scores))
    print("Scores ablated: ", scores_ablated)
    print("Mean score ablated: ", numpy.mean(scores_ablated))


if __name__ == "__main__":

    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 42

    # /usr/local/opt/python@3.10/bin/python3

    #check_score_ablation_recurrent("models/298__OmYURVRc9s214g__4391.npy")
    #check_no_food(path=brain_path)
    #check_place_grid_cell(path=brain_path)
    plot_scores()
    #evolve_brains(seed=seed)
    #check_score("models/13736__eE0Bg2XoQhKWdw__4466.npy")
