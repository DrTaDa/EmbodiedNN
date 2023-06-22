import numpy
import pickle
import nest
import glob
import copy
import sys

from pawn import Agent
from environment import Environment


def mutate_weight(weight, excitatory=True):

    if weight != 0:

        rand = numpy.random.rand()

        if rand < 0.001:
            if numpy.random.rand() < 0.1:
                return 0
            new_weight = numpy.random.uniform(0, 1)
            if not excitatory:
                new_weight = -new_weight
            return new_weight

        elif rand < 0.01:
            if excitatory:
                return numpy.clip(weight + numpy.random.uniform(0, 0.1) - 0.05, 0, 1000)
            else:
                return numpy.clip(weight + numpy.random.uniform(0, 0.1) - 0.05, -1000, 0)

        elif rand < 0.1:
            if excitatory:
                return numpy.clip(weight + numpy.random.uniform(0, 0.01) - 0.005, 0, 1000)
            else:
                return numpy.clip(weight + numpy.random.uniform(0, 0.01) - 0.005, -1000, 0)

    elif weight == 0 and numpy.random.rand() < 0.01:
        if excitatory:
            return numpy.random.uniform(0, 1)
        return numpy.random.uniform(-1, 0)

    return weight


def mutate(child):

    #  Mutate the weights
    for connection in child["connections_exc_exc"]:
        connection["weight"] = mutate_weight(connection["weight"], excitatory=True)

    for connection in child["connections_exc_inh"]:
        connection["weight"] = mutate_weight(connection["weight"], excitatory=True)

    for connection in child["connections_inh_exc"]:
        connection["weight"] = mutate_weight(connection["weight"], excitatory=False)

    for connection in child["connections_inh_inh"]:
        connection["weight"] = mutate_weight(connection["weight"], excitatory=True)

    # Do sensory inputs:
    for i in range(2):
        for connection in child[f"sensory_inputs_{i}"]:
            connection["weight"] = mutate_weight(connection["weight"], excitatory=True)

    if numpy.random.random() > 0.95:
        child["mu_plus"] = numpy.random.uniform(0, 1)
    if numpy.random.random() > 0.95:
        child["mu_minus"] = numpy.random.uniform(0, 1)

    return child


def mate_and_mutate(world_size, sorted_models):
    """Pick two parents and mate them, then mutate the child"""

    with open(numpy.random.choice(sorted_models), "rb") as file:
        parent1 = pickle.load(file)
    child = copy.deepcopy(parent1)

    child = mutate(child)

    # Save to a temporary file
    with open(f"models/tmp.pkl", "wb") as file:
        pickle.dump(child, file)

    return Agent(world_size, position=None, from_file="models/tmp.pkl")


def get_current_generation(max_pop_size=60):

    blob_per_generation = {}
    for blob in glob.glob(f"models/*__*.pkl"):
        generation = int(blob.split("__")[0].split("/")[1])
        if generation not in blob_per_generation:
            blob_per_generation[generation] = []
        blob_per_generation[generation].append(blob)

    if len(blob_per_generation) == 0:
        return 0

    sorted_generations = sorted(blob_per_generation.keys())
    for generation in sorted_generations[::-1]:
        if len(blob_per_generation[generation]) < max_pop_size:
            return generation
        break

    return sorted_generations[-1] + 1


def evolve_brains(seed):

    # set the random seed
    nest.SetKernelStatus({'rng_seed': seed})
    numpy.random.seed(seed)

    nest.ResetKernel()
    nest.set_verbosity(100)

    chunk_size = 300
    vision_distance = chunk_size
    n_chunks = 3
    world_size = n_chunks * chunk_size

    max_pop_size = 100
    max_ngen = 30
    n_test_per_agent = 5
    current_generation = 0

    while current_generation < max_ngen:

        current_generation = get_current_generation(max_pop_size)

        if current_generation > 0:
            sorted_models = sorted(
                glob.glob(f"models/*__*.pkl"),
                key=lambda x: int(x.split("__")[-1].split(".")[0])
            )[::-1]
            # Only keep the best 10 models
            sorted_models = sorted_models[:10]

        nest.ResetKernel()

        if current_generation == 0:
            agent = Agent(world_size)
        else:
            agent = mate_and_mutate(world_size, sorted_models)

        print()
        print(f"Gen {current_generation}, Agent {agent.name}:")

        scores = []
        for e in range(n_test_per_agent):
            agent.reset()
            env = Environment(world_size, vision_distance)
            score = env.test_agent(agent)
            scores.append(score)
            print(f"Epoch {e}, score {score}")

        mean_score = numpy.mean(scores)
        print(f"Mean score: {mean_score}")

        agent.brain.save_network(
            f"models/{current_generation}__{agent.name}__{round(mean_score)}.pkl", scores
        )

        del agent


def plot_scores():

    import matplotlib.pyplot as plt

    scores = []
    gens = range(1 + max([int(file.split("__")[0].split("/")[-1]) for file in glob.glob(f"models/*__*.pkl")]))

    for gen in gens:
        scores.append([])
        for file in glob.glob(f"models/{gen}__*.pkl"):
            score = int(file.split("__")[-1].split(".")[0])
            scores[-1].append(score)

    theo_max = (100 + (80 * 50)) / 0.3
    plt.plot(gens, [theo_max] * len(gens), label="Theoretical max ({})".format(round(theo_max)), linestyle="--", color="black")
    plt.plot(gens, [numpy.mean(s) for s in scores], label="Mean")
    plt.plot(gens, [numpy.max(s) for s in scores], label="Max")
    plt.plot(gens, [numpy.min(s) for s in scores], label="Min")
    plt.fill_between(gens, [numpy.mean(s) - numpy.std(s) for s in scores], [numpy.mean(s) + numpy.std(s) for s in scores], alpha=0.2)

    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.ylim(0, numpy.max([numpy.max(s) for s in scores]) + 1000)

    plt.show()


def check_place_grid_cell(path):

    nest.ResetKernel()
    nest.set_verbosity(100)

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

    nest.ResetKernel()
    nest.set_verbosity(100)

    chunk_size = 300
    vision_distance = chunk_size
    n_chunks = 3
    world_size = n_chunks * chunk_size

    agent = Agent(world_size, position=None, from_file=path)
    agent.reset()

    env = Environment(world_size, vision_distance, n_food=10)
    env.test_agent(agent)


def check_score_ablation_recurrent(path):

    nest.ResetKernel()
    nest.set_verbosity(100)

    chunk_size = 300
    vision_distance = chunk_size
    n_chunks = 3
    world_size = n_chunks * chunk_size

    scores = []
    scores_ablated = []

    for i in range(5):
        nest.ResetKernel()
        nest.set_verbosity(100)
        agent = Agent(world_size, position=None, from_file=path)
        # Remove recurrent connections
        agent.brain.ablate_recurrent_connections()
        agent.reset()
        env = Environment(world_size, vision_distance)
        scores_ablated.append(env.test_agent(agent))
        del agent

    for i in range(5):
        nest.ResetKernel()
        nest.set_verbosity(100)
        agent = Agent(world_size, position=None, from_file=path)
        agent.reset()
        env = Environment(world_size, vision_distance)
        scores.append(env.test_agent(agent))
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

    #brain_path = './models/18__w0Wz7buNPvSekw__2814.pkl'
    #check_score_ablation_recurrent(brain_path)
    #check_no_food(path=brain_path)
    #check_place_grid_cell(path=brain_path)
    #plot_scores()
    evolve_brains(seed=seed)
