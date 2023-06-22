import numpy
import nest
import matplotlib.pyplot as plt
import pprint
import pickle
import networkx as nx


def benchmark_hunger(brain):

    hunger_levels = list(range(0, 101, 10))
    position_food = list(range(0, 9))

    left = numpy.zeros((len(hunger_levels), len(position_food)))
    right = numpy.zeros((len(hunger_levels), len(position_food)))
    forward = numpy.zeros((len(hunger_levels), len(position_food)))
    for h, hunger in enumerate(hunger_levels):
        for p, pos in enumerate(position_food):
            for i in range(10):
                print('hunger: {}, pos: {}, i: {}'.format(hunger, pos, i))
                observation = numpy.zeros(29)
                observation[pos*3 + 1] = 1
                observation[-2] = hunger / 100.
                observation[-1] = 0
                brain.prepare_simulation(observation)
                nest.Simulate(100)
                acts, _ = brain.get_action()
                print(acts)
                left[h][p] += int(acts[0])
                right[h][p] += int(acts[1])
                forward[h][p] += int(acts[2])

    left /= 10
    right /= 10
    forward /= 10

    fig, ax = plt.subplots(3)
    ax[0].imshow(left)
    ax[1].imshow(right)
    ax[2].imshow(forward)

    fig.tight_layout()
    plt.show()


def plot_brain_graph(brain):

    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.Graph()
    for connection in brain["recurrent_connections"]:
        if connection["weight"] != 0:
            G.add_edge(connection["source"], connection["target"], weight=abs(connection["weight"]))

    for connection in brain["sensory_inputs"]:
        if connection["weight"] != 0:
            G.add_edge(connection["source"], connection["target"], weight=abs(connection["weight"]))

    widths = nx.get_edge_attributes(G, 'weight')
    nodelist = G.nodes()

    pos = nx.spiral_layout(G, resolution=0.5)

    nx.draw_networkx_nodes(G, pos,
                           nodelist=nodelist,
                           node_size=1000,
                           node_color='black',
                           alpha=0.7)
    nx.draw_networkx_edges(G, pos,
                           edgelist=widths.keys(),
                           width=list([2 * abs(w) for w in widths.values()]),
                           edge_color='lightblue',
                           alpha=0.8)
    nx.draw_networkx_labels(G, pos=pos,
                            labels=dict(zip(nodelist, nodelist)),
                            font_color='white')
    plt.show()


def brain_graph_SSC_centrality(brain, threshold_weight=0.):
    """Plot the strongly connected components"""

    G = nx.DiGraph()

    for connection in brain["recurrent_connections"]:
        if abs(connection["weight"]) > threshold_weight:
            G.add_edge(connection["source"], connection["target"])

    for connection in brain["sensory_inputs"]:
        if abs(connection["weight"]) > threshold_weight:
            G.add_edge(connection["source"], connection["target"])

    for centrality_func in [nx.degree_centrality, nx.betweenness_centrality, nx.closeness_centrality, nx.eigenvector_centrality]:
        centrality = centrality_func(G)
        print(str(centrality_func))
        print("Mean centrality outputs: ", numpy.mean([centrality[id_] for id_ in brain["output_neurons_id"]]))
        print("Mean centrality: ", numpy.mean(list(centrality.values())))
        print()


def plot_brain_connection_matrix(brain):
    import matplotlib.pyplot as plt

    connections = numpy.zeros((brain["n_neurons"][0] + 29, brain["n_neurons"][0] + 29))
    for connection in brain["recurrent_connections"]:
        connections[connection["source"] - 1, connection["target"] - 1] = connection["weight"]

    for connection in brain["sensory_inputs"]:
        connections[connection["source"] - 1, connection["target"] - 1] = connection["weight"]

    fig, ax1 = plt.subplots(1)
    fig.suptitle('Brain connection matrix')
    ax1.imshow(connections)

    # show colormap on the right
    fig.colorbar(ax1.imshow(connections), ax=ax1)

    # Highlight output neurons
    for i, id_ in enumerate(brain["output_neurons_id"]):
        ax1.axvline(id_ - 1, color="red", alpha=0.5)
        if i == 0:
            ax1.text(id_ - 1.5, -1, "Left", color="red", rotation=90)
        elif i == 1:
            ax1.text(id_ - 1.5, -1, "Right", color="red", rotation=90)
        elif i == 2:
            ax1.text(id_ - 1.5, -1, "Forward", color="red", rotation=90)

    ax1.set_xlabel("Input neurons")
    ax1.set_ylabel("Output neurons")

    plt.show()


def plot_brain_statistics(file):

    random_brains = [pickle.load(open("tmp_random_brain_{}.pkl".format(i), "rb")) for i in range(10)]

    # set the random seed
    nest.SetKernelStatus({'rng_seed': 42})
    numpy.random.seed(42)

    nest.ResetKernel()
    nest.set_verbosity(100)

    chunk_size = 300
    vision_distance = chunk_size

    n_chunks = 5
    world_size = n_chunks * chunk_size

    with open(file, "rb") as fp:
        brain = pickle.load(fp)

    import pprint
    pprint.pprint(brain)
    plot_brain_graph(brain)
    plot_brain_connection_matrix(brain)
    brain_graph_SSC_centrality(brain)

    # Print average number of connections per neuron
    n_excitatory_connections = 0
    n_inhibitory_connections = 0
    for connection in brain["recurrent_connections"]:
        if connection["weight"] > 0:
            n_excitatory_connections += 1
        elif connection["weight"] < 0:
            n_inhibitory_connections += 1

    n_excitatory_connections_random = 0
    n_inhibitory_connections_random = 0
    for br in random_brains:
        for connection in br["recurrent_connections"]:
            if connection["weight"] > 0:
                n_excitatory_connections_random += 1
            elif connection["weight"] < 0:
                n_inhibitory_connections_random += 1

    print("Average number of excitatory connections per neuron: ", n_excitatory_connections / brain["n_neurons"][0])
    print("Average number of inhibitory connections per neuron: ", n_inhibitory_connections / brain["n_neurons"][0])
    print("Average number of excitatory connections per neuron (random): ", n_excitatory_connections_random / (brain["n_neurons"][0] * len(random_brains)))
    print("Average number of inhibitory connections per neuron (random): ", n_inhibitory_connections_random / (brain["n_neurons"][0] * len(random_brains)))
    print()

    # Print average number of connections per sensory input
    n_excitatory_connections = 0
    n_inhibitory_connections = 0
    for connection in brain["sensory_inputs"]:
        if connection["weight"] > 0:
            n_excitatory_connections += 1
        elif connection["weight"] < 0:
            n_inhibitory_connections += 1

    n_excitatory_connections_random = 0
    n_inhibitory_connections_random = 0
    for br in random_brains:
        for connection in br["sensory_inputs"]:
            if connection["weight"] > 0:
                n_excitatory_connections_random += 1
            elif connection["weight"] < 0:
                n_inhibitory_connections_random += 1

    print("Average number of excitatory connections per sensory input: ", n_excitatory_connections / 20)
    print("Average number of inhibitory connections per sensory input: ", n_inhibitory_connections / 20)
    print("Average number of excitatory connections per sensory input (random): ", n_excitatory_connections_random / (20 * len(random_brains)))
    print("Average number of inhibitory connections per sensory input (random): ", n_inhibitory_connections_random / (20 * len(random_brains)))
    print()

    # Plot the distribution of the weights
    weights = [connection["weight"] for connection in brain["recurrent_connections"]]
    weights = numpy.array(weights)
    weights_random = []
    for br in random_brains:
        for connection in br["recurrent_connections"]:
            weights_random.append(connection["weight"])
    plt.title("Distribution of the weights")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.hist(weights_random, bins=100, alpha=0.5, label="Random", density=True)
    plt.hist(weights, bins=100, density=True)
    plt.show()

    # Plot the distribution of the sensory inputs
    weights = [connection["weight"] for connection in brain["sensory_inputs"]]
    weights = numpy.array(weights)
    weights_random = []
    for br in random_brains:
        for connection in br["sensory_inputs"]:
            weights_random.append(connection["weight"])
    plt.title("Distribution of the sensory inputs")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.hist(weights_random, bins=100, alpha=0.5, label="Random", density=True)
    plt.hist(weights, bins=100, density=True)
    plt.show()

    # Plot the distribution of the number of connections per neuron
    n_connections_per_neuron = {}
    for connection in brain["recurrent_connections"]:
        if connection["source"] not in n_connections_per_neuron:
            n_connections_per_neuron[connection["source"]] = 0
        if connection["weight"] != 0:
            n_connections_per_neuron[connection["source"]] += 1

    n_connections_per_neuron_random = []
    for br in random_brains:
        n_connections_per_neuron_random.append({})
        for connection in br["recurrent_connections"]:
            if connection["source"] not in n_connections_per_neuron_random[-1]:
                n_connections_per_neuron_random[-1][connection["source"]] = 0
            if connection["weight"] != 0:
                n_connections_per_neuron_random[-1][connection["source"]] += 1

    n_connections_per_neuron = numpy.array(list(n_connections_per_neuron.values()))
    n_connections_per_neuron_random = numpy.array([list(n_connections_per_neuron_random[i].values()) for i in range(len(n_connections_per_neuron_random))])
    plt.title("Distribution of the number of connections per neuron")
    plt.xlabel("Number of connections")
    plt.ylabel("Number of neurons")
    plt.hist(n_connections_per_neuron_random.flatten(), bins=10, alpha=0.5, label="Random", density=True)
    plt.hist(n_connections_per_neuron, bins=10, density=True)
    plt.show()

    # Plot the distribution of the number of connections per sensory input
    n_connections_per_neuron = {}
    for connection in brain["sensory_inputs"]:
        if connection["source"] not in n_connections_per_neuron:
            n_connections_per_neuron[connection["source"]] = 0
        if connection["weight"] != 0:
            n_connections_per_neuron[connection["source"]] += 1

    n_connections_per_neuron_random = []
    for br in random_brains:
        n_connections_per_neuron_random.append({})
        for connection in br["sensory_inputs"]:
            if connection["source"] not in n_connections_per_neuron_random[-1]:
                n_connections_per_neuron_random[-1][connection["source"]] = 0
            if connection["weight"] != 0:
                n_connections_per_neuron_random[-1][connection["source"]] += 1

    n_connections_per_neuron_random = numpy.array([list(n_connections_per_neuron_random[i].values()) for i in range(len(n_connections_per_neuron_random))])
    n_connections_per_neuron = numpy.array(list(n_connections_per_neuron.values()))
    plt.title("Distribution of the number of connections per sensory input")
    plt.xlabel("Number of connections")
    plt.ylabel("Number of neurons")
    plt.hist(n_connections_per_neuron_random.flatten(), bins=10, alpha=0.5, label="Random", density=True)
    plt.hist(n_connections_per_neuron, bins=10, density=True)
    plt.show()


def main():

    from brain import Brain

    nest.set_verbosity(100)

    brain_path = './models/18__w0Wz7buNPvSekw__2814.pkl'
    brain = Brain(n_sensory_inputs=29, n_output_neurons=3, from_file=brain_path)

    # random_brains = [Brain(n_sensory_inputs=29, n_output_neurons=3) for _ in range(10)]
    # for i, brain in enumerate(random_brains):
    #     brain.save_network("tmp_random_brain_{}.pkl".format(i), 0)
    # plot_brain_statistics(brain_path)

    benchmark_hunger(brain)


if __name__ == '__main__':
    main()
