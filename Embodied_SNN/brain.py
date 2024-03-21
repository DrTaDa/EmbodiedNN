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
        self.max_frequency = 100  # Hz

        self.fraction_inhibitory_neurons = 0.25
        self.n_hidden_neurons = 50
        self.n_inhibitory_neurons = int(self.fraction_inhibitory_neurons * self.n_hidden_neurons)
        self.n_excitatory_neurons = self.n_hidden_neurons - self.n_inhibitory_neurons

        self.n_sensory_inputs = n_sensory_inputs
        self.n_output_neurons = n_output_neurons

        self.sensory_inputs = nest.Create('poisson_generator', n_sensory_inputs)
        self.output_brain_regions = None
        self.output_neurons_id = []

        self.networks = []
        #self.voltmeters = []

        self.mu_plus = numpy.random.uniform(0, 1)
        self.mu_minus = numpy.random.uniform(0, 1)
        self.Wmax = 10

        # This is to keep in memory the last spikes to control the excitability on
        # the scale of several time steps (max_history_length time steps)
        self.spike_history = []
        self.max_history_length = 10

        self.hunger_when_eating = []
        self.thirst_when_drinking = []
        self.cause_of_death = []
        self.actions_history = []

        if from_file is None:
            self.create_network()
        else:
            self.load_network(from_file)

        # Recode spikes from output neurons
        self.spike_recorders = []
        for i, network in enumerate(self.networks):
            spike_recorder = nest.Create("spike_recorder")
            nest.Connect(network, spike_recorder)
            self.spike_recorders.append(spike_recorder)
        self.spike_recorder_output = nest.Create("spike_recorder")
        nest.Connect(self.output_brain_regions, self.spike_recorder_output)

        # Add a small noise generator for the whole network
        self.global_noise_generator = nest.Create("poisson_generator", 1, {"rate": 5.0})
        nest.Connect(self.global_noise_generator, self.networks[0])

    @property
    def syn_spec(self):
        return {
            "synapse_model": "stdp_synapse",
            "mu_plus": self.mu_plus,
            "mu_minus": self.mu_minus,
            "Wmax": 1,
        }

    @property
    def conn_spec(self):
        return {"rule": "all_to_all"}

    def set_E_L(self, E_L=-55.1):
        for i in range(len(self.networks)):
            for j in range(len(self.networks[i])):
                self.networks[i][j].set(E_L=E_L)

    def randomly_connect(self, nodes1, nodes2, excitatory=True, sensory_connection=False):

        if excitatory and not sensory_connection:
            syn_spec = self.syn_spec
            syn_spec["Wmax"] = 1
            syn_spec["weight"] = 1

            nest.Connect(
                nodes1,
                nodes2,
                conn_spec=self.conn_spec,
                syn_spec=syn_spec
            )

        else:
            nest.Connect(
                nodes1,
                nodes2,
                conn_spec={"rule": "all_to_all"},
                syn_spec={"weight": 0}
            )

        p_connection = 0.1
        connections = nest.GetConnections(source=nodes1, target=nodes2)

        for k in range(len(connections)):
            if numpy.random.random() < p_connection:
                if excitatory:
                    nest.SetStatus(connections[k], {"weight": numpy.random.uniform(0, self.Wmax)})
                else:
                    nest.SetStatus(connections[k], {"weight": numpy.random.uniform(-self.Wmax, 0)})
            else:
                nest.SetStatus(connections[k], {"weight": 0})

    def create_network(self):
        """Create a random brain structure"""

        self.networks.append(nest.Create("iaf_psc_alpha", self.n_excitatory_neurons))
        self.networks.append(nest.Create("iaf_psc_alpha", self.n_inhibitory_neurons))

        #voltmeter = nest.Create("voltmeter")
        #nest.Connect(voltmeter, self.networks[-1])
        #self.voltmeters.append(voltmeter)

        # Create random connections between networks
        self.randomly_connect(self.networks[0], self.networks[0], excitatory=True)
        self.randomly_connect(self.networks[0], self.networks[1], excitatory=True)
        self.randomly_connect(self.networks[1], self.networks[0], excitatory=False)
        self.randomly_connect(self.networks[1], self.networks[1], excitatory=True)

        # Create random connections between sensory inputs and networks
        self.randomly_connect(self.sensory_inputs, self.networks[0], excitatory=True, sensory_connection=True)
        self.randomly_connect(self.sensory_inputs, self.networks[1], excitatory=True, sensory_connection=True)

        # Select a random subset of neurons to be output neurons
        self.output_brain_regions = self.networks[0]
        indexes = numpy.random.choice(len(self.output_brain_regions), self.n_output_neurons, replace=False)
        self.output_neurons_id = [self.output_brain_regions[id_].global_id for id_ in indexes]

        self.set_E_L()

    def prepare_simulation(self, observation):
        """Prepare the simulation by setting the frequency of the sensory inputs."""

        # print()
        # print(observation)

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

        if sum(frequencies) == 0:
            return 0

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

    def homeostatic_frequency_rule(self):
        for i in range(len(self.networks)):
            for j in range(len(self.networks[i])):
                freq = sum([self.spike_history[t][self.networks[i][j].global_id] if self.networks[i][j].global_id in
                            self.spike_history[t] else 0 for t in range(len(self.spike_history))])
                if freq < 1 and self.networks[i][j].get("E_L") < -55.02:
                    self.networks[i][j].set(E_L=self.networks[i][j].get("E_L") + 0.02)
                elif freq > 20:
                    self.networks[i][j].set(E_L=self.networks[i][j].get("E_L") - 0.02)

    def get_action(self):
        """Select an action based on the frequency output of the network since the last action."""

        # Get the frequency of each output neurons
        spike_trains = self.spike_recorder_output.get("events")
        events = spike_trains["senders"].tolist()
        frequencies = {}
        ids = set(events)
        for id_ in ids:
            frequencies[id_] = events.count(id_)

        actions = [frequencies[id_] if (id_ in frequencies) else 0 for id_ in self.output_neurons_id]

        # print(list(si.get("rate") for si in self.sensory_inputs))
        # dmm = self.voltmeters[-1].get()
        # for i in range(len(self.networks[-1])):
        #     vms = dmm["events"]["V_m"][i::len(self.networks[-1])]
        #     ts = dmm["events"]["times"][i::len(self.networks[-1])]
        #     plt.plot(ts, vms)
        # #plt.ylim(-75, -50)
        # plt.xlabel("Time (ms)")
        # plt.ylabel("Voltage (mV)")
        # plt.show()

        # Homeostatic frequency rule for individual neurons
        self.spike_history.append(frequencies)
        if len(self.spike_history) > self.max_history_length:
            self.spike_history.pop(0)

        self.homeostatic_frequency_rule()

        #print(actions)
        #print(f"Mean E_L: {numpy.mean([n.get('E_L') for n in self.networks[0]])} mV")

        spike_trains = {'senders': [], 'times': []}
        for i in range(len(self.networks)):
            spike_trains['senders'].extend(self.spike_recorders[i].get("events")["senders"].tolist())
            spike_trains['times'].extend(self.spike_recorders[i].get("events")["times"].tolist())

        # Reset the spike recorders and the voltmeters
        for i in range(len(self.networks)):
            self.spike_recorders[i].set(n_events=0)
        self.spike_recorder_output.set(n_events=0)
        #for i in range(len(self.networks)):
        #    self.voltmeters[i].set(n_events=0)

        return actions, spike_trains

    def save_network(self, path, scores):
        """Save the network to a file.
        Needs to build a dictionary with relevant network information:
            - number of networks
            - number of neurons in each brain region
            - source, target and weight of all connections
            - connection rule
            - output brain region and neurons
        """

        network = {"n_brain_regions": len(self.networks),
                   "n_neurons": [len(brain_region) for brain_region in self.networks],
                   "output_neurons_id": self.output_neurons_id, "scores": scores,
                   "hunger_when_eating": self.hunger_when_eating,
                   "thirst_when_drinking": self.thirst_when_drinking,
                   "cause of death": self.cause_of_death,
                   "actions_history": self.actions_history,
                   "mu_plus": self.mu_plus,
                   "mu_minus": self.mu_minus,
                   "Wmax": self.Wmax}

        connections_exc_exc = nest.GetConnections(source=self.networks[0], target=self.networks[0])
        connections_exc_inh = nest.GetConnections(source=self.networks[0], target=self.networks[1])
        connections_inh_exc = nest.GetConnections(source=self.networks[1], target=self.networks[0])
        connections_inh_inh = nest.GetConnections(source=self.networks[1], target=self.networks[1])

        network["connections_exc_exc"] = []
        for connection in connections_exc_exc:
            network["connections_exc_exc"].append({
                "source": connection.source,
                "target": connection.target,
                "weight": connection.weight
            })

        network["connections_exc_inh"] = []
        for connection in connections_exc_inh:
            network["connections_exc_inh"].append({
                "source": connection.source,
                "target": connection.target,
                "weight": connection.weight
            })

        network["connections_inh_exc"] = []
        for connection in connections_inh_exc:
            network["connections_inh_exc"].append({
                "source": connection.source,
                "target": connection.target,
                "weight": connection.weight
            })

        network["connections_inh_inh"] = []
        for connection in connections_inh_inh:
            network["connections_inh_inh"].append({
                "source": connection.source,
                "target": connection.target,
                "weight": connection.weight
            })

        # Add connections between sensory inputs and networks
        for i in range(len(self.networks)):
            connections = nest.GetConnections(source=self.sensory_inputs, target=self.networks[i])
            network[f"sensory_inputs_{i}"] = []
            for connection in connections:
                network[f"sensory_inputs_{i}"].append({
                    "source": connection.source,
                    "target": connection.target,
                    "weight": connection.weight
                })

        with open(path, "wb") as f:
            pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)

    def load_network(self, file):
        """Load a network from a file."""

        brain_data = pickle.load(open(file, "rb"))

        self.mu_plus = brain_data["mu_plus"]
        self.mu_minus = brain_data["mu_minus"]
        self.Wmax = brain_data["Wmax"]

        # Create the brain regions
        self.networks = []
        for i in range(brain_data["n_brain_regions"]):
            self.networks.append(nest.Create("iaf_psc_alpha", brain_data["n_neurons"][i]))
            #voltmeter = nest.Create("voltmeter")
            #nest.Connect(voltmeter, self.networks[-1])
            #self.voltmeters.append(voltmeter)

        tmp_neuron_dict = {}
        for i in range(len(self.networks)):
            for j in range(len(self.networks[i])):
                tmp_neuron_dict[self.networks[i][j].global_id] = self.networks[i][j]

        # Create the connections between sensory inputs and brain regions
        for i in range(len(self.networks)):
            for connection in brain_data[f"sensory_inputs_{i}"]:
                source = next(
                    (x for x in self.sensory_inputs if int(x.global_id) == int(connection["source"]))
                )
                target = tmp_neuron_dict[int(connection["target"])]
                nest.Connect(
                    source, target,
                    syn_spec={"weight": float(connection["weight"])}
                )

        # Create the connections between brain regions
        for connection in brain_data["connections_exc_exc"]:
            source = tmp_neuron_dict[int(connection["source"])]
            target = tmp_neuron_dict[int(connection["target"])]
            syn_spec = self.syn_spec
            syn_spec["weight"] = float(connection["weight"])
            nest.Connect(
                source, target,
                syn_spec=syn_spec
            )

        for connection in brain_data["connections_exc_inh"]:
            source = tmp_neuron_dict[int(connection["source"])]
            target = tmp_neuron_dict[int(connection["target"])]
            syn_spec = self.syn_spec
            syn_spec["weight"] = float(connection["weight"])
            nest.Connect(
                source, target,
                syn_spec=syn_spec
            )

        for connection in brain_data["connections_inh_exc"]:
            source = tmp_neuron_dict[int(connection["source"])]
            target = tmp_neuron_dict[int(connection["target"])]
            # WARNING: This is a hack to avoid the problem of inhibitory STDP
            nest.Connect(
                source, target,
                syn_spec={"weight": float(connection["weight"])}
            )

        for connection in brain_data["connections_inh_inh"]:
            source = tmp_neuron_dict[int(connection["source"])]
            target = tmp_neuron_dict[int(connection["target"])]
            syn_spec = self.syn_spec
            syn_spec["weight"] = float(connection["weight"])
            nest.Connect(
                source, target,
                syn_spec=syn_spec
            )

        # Set the output neurons
        self.output_neurons_id = brain_data["output_neurons_id"]
        self.output_brain_regions = None
        for brain_region in self.networks:
            if any(brain_region[i].global_id in self.output_neurons_id for i in range(len(brain_region))):
                self.output_brain_regions = brain_region
                break

        self.set_E_L()

    def ablate_recurrent_connections(self):
        for i in range(len(self.networks)):
            connections = nest.GetConnections(source=self.networks[i], target=self.networks[i])
            rmv_conn = 0
            for connection in connections:
                if connection.source == connection.target and connection.weight != 0.0:
                    rmv_conn += 1
                    nest.SetStatus(connection, {"weight": 0.0})
