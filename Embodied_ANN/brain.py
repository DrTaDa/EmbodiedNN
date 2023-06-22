import numpy


def step(x):
    return numpy.heaviside(x, 1)


def double_relu(x):
    return numpy.clip(x, -1, 1)


def glorot_uniform(n_in, n_out):
    limit = numpy.sqrt(6 / (n_in + n_out))
    weights = limit * ((numpy.random.random((n_in, n_out)) * 2.) - 1)
    biases = limit * ((numpy.random.random(n_out) * 2.) - 1)
    return weights, biases


class Brain():

    layer_names = [
        "layer_hidden", "biases_hidden", "layer_mouv",
        "biases_mouv", "layer_mem", "biases_mem"
    ] # "layer_act", "biases_act"

    def __init__(
        self,
        n_vision_rays=13,
        length_vision_vector=3,
        n_other_inputs=2,
        n_actions=0,
        n_memory=20,
        n_hidden=100,
    ):

        self.input_size = (length_vision_vector * n_vision_rays) + n_other_inputs  + n_memory

        self.n_hidden = n_hidden
        self.n_memory = n_memory
        self.n_actions = n_actions

        self.layer_hidden, self.biases_hidden = glorot_uniform(self.input_size, self.n_hidden)
        self.activ_hidden = step

        self.layer_mouv, self.biases_mouv = glorot_uniform(self.n_hidden, 2)
        self.activ_mouv = double_relu

        # self.layer_act, self.biases_act = glorot_uniform(self.n_hidden, self.n_actions)
        # self.activ_act = step

        self.layer_mem, self.biases_mem = glorot_uniform(self.n_hidden, self.n_memory)
        self.activ_mem = step

        self.cause_of_death = []
        self.hunger_when_eating = []
        self.thirst_when_drinking = []

    def evaluate(self, input):

        output_hidden = self.activ_hidden(input.dot(self.layer_hidden) + self.biases_hidden)
        output_mov = self.activ_mouv(output_hidden.dot(self.layer_mouv) + self.biases_mouv)
        output_mem = self.activ_mem(output_hidden.dot(self.layer_mem) + self.biases_mem)

        return output_mov, output_mem

    def copy_weights_and_mutate_point(self, other_brain):

        n_point_mutations = 0

        for layer_name in self.layer_names:

            if other_brain is not None:

                new_weights_a = getattr(self, layer_name)
                new_weights_b = getattr(other_brain, layer_name)

                if isinstance(new_weights_a[0], float):

                    split_idx = numpy.random.randint(0, len(new_weights_a))
                    if numpy.random.random() > 0.5:
                        new_weights = numpy.concatenate(
                            (new_weights_a[:split_idx], new_weights_b[split_idx:]), axis=0)
                    else:
                        new_weights = numpy.concatenate(
                            (new_weights_b[:split_idx], new_weights_a[split_idx:]), axis=0)

                else:

                    new_weights = []
                    for i, w in enumerate(new_weights_a):
                        split_idx = numpy.random.randint(0, len(w))
                        if numpy.random.random() > 0.5:
                            new_weights.append(numpy.concatenate(
                                (new_weights_a[i][:split_idx], new_weights_b[i][split_idx:]), axis=0))
                        else:
                            new_weights.append(numpy.concatenate(
                                (new_weights_b[i][:split_idx], new_weights_a[i][split_idx:]), axis=0))

            else:
                new_weights = getattr(self, layer_name)

            for i, w in enumerate(new_weights):

                if isinstance(w, float):
                    tmp_rand = numpy.random.random()
                    if tmp_rand > 0.99:
                        new_weights[i] = numpy.random.uniform(-0.5, 0.5)
                        n_point_mutations += 1
                    # elif tmp_rand > 0.95:
                    #     new_weights[i] += numpy.random.uniform(-0.1, 0.1)
                else:
                    for j, w2 in enumerate(w):
                        tmp_rand = numpy.random.random()
                        if tmp_rand > 0.99:
                            new_weights[i][j] = numpy.random.uniform(-0.5, 0.5)
                            n_point_mutations += 1
                        # elif tmp_rand > 0.95:
                        #     new_weights[i] += numpy.random.uniform(-0.1, 0.1)
            setattr(self, layer_name, new_weights)

        print(f"N point mutations: {n_point_mutations}")

        if numpy.random.random() > 0.9:
            self.layer_hidden = numpy.array(
                [list(ws) + [0] for i, ws in enumerate(self.layer_hidden)])
            self.biases_hidden = numpy.array(
                list(self.biases_hidden) + [0])
            self.layer_mouv = numpy.concatenate(
                (self.layer_mouv, numpy.zeros((1, 2))), axis=0
            )
            self.layer_mem = numpy.vstack([self.layer_mem, numpy.array([0] * self.n_memory)])
            self.n_hidden += 1
            print(f"Increase hidden neuron")

        if numpy.random.random() > 0.9:
            self.layer_mem = numpy.array(
                [list(ws) + [0] for i, ws in enumerate(self.layer_mem)])
            self.biases_mem = numpy.array(
                list(self.biases_mem) + [numpy.random.uniform(-1, 1)])
            self.layer_hidden = numpy.vstack([self.layer_hidden, numpy.array([0] * self.n_hidden)])
            self.n_memory += 1
            print(f"Increase memory neuron")

    def save(self, name, score):

        to_save = []
        for layer_name in self.layer_names:
            to_save.append(getattr(self, layer_name))
        to_save.append(self.cause_of_death)
        to_save.append(self.hunger_when_eating)
        to_save.append(self.thirst_when_drinking)
        numpy.save(f"./models/{name}__{score}.npy", to_save)

    def load(self, path):

        new_weights = numpy.load(path, allow_pickle=True)
        for i, layer_name in enumerate(self.layer_names):
            setattr(self, layer_name, new_weights[i])

            if layer_name == "biases_hidden":
                self.n_hidden = len(getattr(self, layer_name))
            if layer_name == "biases_mem":
                self.n_memory = len(getattr(self, layer_name))
