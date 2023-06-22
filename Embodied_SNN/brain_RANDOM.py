import numpy
import matplotlib.pyplot as plt
import nest
import pickle
import pandas as pd

LIST_OF_RULES = [
    "all_to_all",
    "fixed_indegree",
    "fixed_outdegree",
    "pairwise_bernoulli"
]

CONNECTION_SPECS = {
    "all_to_all": {},
    "fixed_indegree": {'indegree': 10},
    "fixed_outdegree": {'outdegree': 10},
    "pairwise_bernoulli": {'p': 0.1}
}


class Brain(object):

    def __init__(self, n_sensory_inputs=13, n_output_neurons=4, from_file=None):

        self.min_frequency = 0  # Hz
        self.max_frequency = 80000  # Hz

        self.target_frequency = 50  # Hz per neuron

        self.n_sensory_inputs = n_sensory_inputs
        self.n_output_neurons = n_output_neurons

        self.sensory_inputs = nest.Create('poisson_generator', n_sensory_inputs)
        self.output_brain_regions = None
        self.output_neurons_id = []

        self.min_n_networks = 1
        self.max_n_networks = 4

        self.min_n_neurons = 5
        self.max_n_neurons = 51

        self.min_weight = -1
        self.max_weight = 1

        self.networks = []
        self.voltmeters = []

        if from_file is None:
            self.create_network()
            self.mean_noises = [350] * len(self.networks)
        else:
            self.load_network(from_file)

        # Add a permanent noise source to offset the baseline activity
        self.noise_generators = []
        for i, network in enumerate(self.networks):
            noise_generator = nest.Create("noise_generator", 1, {"mean": self.mean_noises[i], "std": 10})
            nest.Connect(noise_generator, network)
            self.noise_generators.append(noise_generator)

        # Recode spikes from output neurons
        self.spike_recorders = []
        for i, network in enumerate(self.networks):
            spike_recorder = nest.Create("spike_recorder")
            nest.Connect(network, spike_recorder)
            self.spike_recorders.append(spike_recorder)
        self.spike_recorder_output = nest.Create("spike_recorder")
        nest.Connect(self.output_brain_regions, self.spike_recorder_output)

    def draw_random_rule(self):
        return numpy.random.choice(LIST_OF_RULES)

    def draw_random_delay(self):
        return numpy.random.uniform(0.1, 10)

    def randomly_connect(self, nodes1, nodes2, sensory_connection=False):

        rule = self.draw_random_rule()
        delay = self.draw_random_delay()
        nest.Connect(
            nodes1,
            nodes2,
            conn_spec={"rule": rule, **CONNECTION_SPECS[rule]},
            syn_spec={'weight': 0, 'delay': delay}
        )

        if sensory_connection:
            exc_inh_multiplier = 1
        else:
            exc_inh_multiplier = numpy.random.choice([-1, 1], p=[0.25, 0.75])

        connections = nest.GetConnections(source=nodes1, target=nodes2)
        for k in range(len(connections)):
            nest.SetStatus(connections[k], {
                "weight": exc_inh_multiplier * numpy.random.uniform(0, self.max_weight)})

    def create_network(self):
        """Create a random brain structure"""

        n_networks = numpy.random.randint(self.min_n_networks, self.max_n_networks)

        for i in range(n_networks):
            n_neurons = numpy.random.randint(self.min_n_neurons, self.max_n_neurons)
            self.networks.append(nest.Create("iaf_psc_alpha", n_neurons))
            voltmeter = nest.Create("voltmeter")
            nest.Connect(voltmeter, self.networks[-1])
            self.voltmeters.append(voltmeter)

        # Create random connections between networks
        is_connection_to_sensory = False
        for i in range(len(self.networks)):
            for j in range(len(self.networks)):
                if numpy.random.choice([True, False]):
                    self.randomly_connect(self.networks[i], self.networks[j])

            # Create random connections between sensory inputs and networks
            if numpy.random.choice([True, False]):
                self.randomly_connect(self.sensory_inputs, self.networks[i])
                is_connection_to_sensory = True

        if not is_connection_to_sensory:
            self.randomly_connect(self.sensory_inputs,
                                  self.networks[numpy.random.randint(0, len(self.networks))])

        # Select a random subset of neurons to be output neurons
        id_brain_regions = numpy.random.randint(0, len(self.networks))
        self.output_brain_regions = self.networks[id_brain_regions]
        indexes = numpy.random.choice(len(self.output_brain_regions), self.n_output_neurons, replace=False)
        self.output_neurons_id = [self.output_brain_regions[id_].global_id for id_ in indexes]

    def prepare_simulation(self, observation):
        """Prepare the simulation by setting the frequency of the sensory inputs."""

        # Set the frequency of the sensory inputs
        for i, obser in enumerate(observation):

            if obser:
                frequency = obser * (self.max_frequency - self.min_frequency) + self.min_frequency
            else:
                frequency = self.min_frequency

            self.sensory_inputs[i].set(rate=frequency)

    def select_action(self, frequencies):
        """If a frequency is above the others, select it, otherwise, select an action at random
        amongst the ones that have the higher frequency."""

        # Select the action with the highest frequency
        action = numpy.argmax(frequencies)

        # If the highest frequency is not above the others, select an action at random
        # amongst the ones that have the higher frequency
        for i, freq in enumerate(frequencies):
            if i != action and freq == frequencies[action]:
                action = numpy.random.choice(
                    [j for j, x in enumerate(frequencies) if x == frequencies[action]]
                )
                return action

        return action

    def get_action(self):
        """Select an action based on the frequency output of the network since the last action."""

        # Get the frequency of each output neurons
        events = self.spike_recorder_output.get("events")["senders"].tolist()
        frequencies = [events.count(id_) for id_ in self.output_neurons_id]

        # Select the action with the highest frequency
        action = self.select_action(frequencies)
        #print(frequencies, sum(frequencies), action)

        # dmm = self.voltmeters[-1].get()
        # for i in range(len(self.networks[-1])):
        #     vms = dmm["events"]["V_m"][i::len(self.networks[-1])]
        #     ts = dmm["events"]["times"][i::len(self.networks[-1])]
        #     plt.plot(ts, vms)
        # #plt.ylim(-75, -50)
        # plt.xlabel("Time (ms)")
        # plt.ylabel("Voltage (mV)")
        # plt.show()

        # Adjust the noise level for each brain region
        for i in range(len(self.networks)):
            spikes = self.spike_recorder_output.get("events")
            frequency = len(spikes["senders"].tolist()) / len(self.networks[i])
            #print(f"Mean frequency: {frequency * 10} Hz for brain region {i}")
            delta_target = (frequency * 10) - self.target_frequency  # 10 times 100ms = 1s
            noise_delta = 5 if delta_target < 0 else -5
            self.noise_generators[i].set(mean=self.noise_generators[i].get("mean") + noise_delta)

        # Reset the spike recorders and the voltmeters
        for i in range(len(self.networks)):
            self.spike_recorders[i].set(n_events=0)
        self.spike_recorder_output.set(n_events=0)
        for i in range(len(self.networks)):
            self.voltmeters[i].set(n_events=0)

        return action

    def save_network(self, path):
        """Save the network to a file.
        Needs to build a dictionary with relevant network information:
            - number of brain regions
            - number of neurons in each brain region
            - source, target and weight of all connections
            - connection rule
            - mean noise level
            - output brain region and neurons
        """

        network = {"n_brain_regions": len(self.networks),
                   "n_neurons": [len(brain_region) for brain_region in self.networks],
                   "mean_noises": [generator.get("mean") for generator in self.noise_generators],
                   "output_neurons_id": self.output_neurons_id, "connections": {}}

        for i in range(len(self.networks)):
            for j in range(len(self.networks)):
                connections = nest.GetConnections(source=self.networks[i], target=self.networks[j])
                if len(connections) > 0:
                    network[f"connection_{i}_{j}"] = pd.DataFrame({
                        "source": connections.source,
                        "target": connections.target,
                        "weight": connections.weight
                    })

        # Add connections between sensory inputs and networks
        for i in range(len(self.networks)):
            connections = nest.GetConnections(source=self.sensory_inputs, target=self.networks[i])
            if len(connections) > 0:
                network[f"sensory_inputs_{i}"] = pd.DataFrame({
                    "source": connections.source,
                    "target": connections.target,
                    "weight": connections.weight
                })

        with open(path, "wb") as f:
            pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)

    def load_network(self, file):
        """Load a network from a file."""

        brain_data = pickle.load(open(file, "rb"))

        # Create the brain regions
        self.networks = []
        for i in range(brain_data["n_brain_regions"]):
            self.networks.append(nest.Create("iaf_psc_alpha", brain_data["n_neurons"][i]))
            voltmeter = nest.Create("voltmeter")
            nest.Connect(voltmeter, self.networks[-1])
            self.voltmeters.append(voltmeter)

        # Create the connections between sensory inputs and brain regions
        for i in range(len(self.networks)):
            if f"sensory_inputs_{i}" in brain_data:
                for _, connection in brain_data[f"sensory_inputs_{i}"].iterrows():
                    source = next(
                        (x for x in self.sensory_inputs if int(x.global_id) == int(connection["source"]))
                    )
                    target = next(
                        (x for x in self.networks[i] if int(x.global_id) == int(connection["target"]))
                    )
                    nest.Connect(
                        source, target,
                        syn_spec={"weight": float(connection["weight"])}
                    )

        # Create the connections between brain regions
        for i in range(len(self.networks)):
            for j in range(len(self.networks)):
                if f"{i}_{j}" in brain_data:
                    for _, connection in brain_data[f"{i}_{j}"].iterrows():

                        source = next(
                            (x for x in self.networks[i] if int(x.global_id) == int(connection["source"]))
                        )
                        target = next(
                            (x for x in self.networks[j] if int(x.global_id) == int(connection["target"]))
                        )
                        nest.Connect(
                            source, target,
                            syn_spec={"weight": float(connection["weight"])}
                        )

        # Set the output neurons
        self.output_neurons_id = brain_data["output_neurons_id"]
        self.output_brain_regions = None
        for brain_region in self.networks:
            if any(brain_region[i].global_id in self.output_neurons_id for i in range(len(brain_region))):
                self.output_brain_regions = brain_region
                break

        self.mean_noises = brain_data["mean_noises"]
