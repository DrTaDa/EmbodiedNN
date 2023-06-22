
### 31/03/2023
Embodied reinforcement learning is the future !
I need to setup a simulation and use NEST for the agent.
The metric for success will be the survival of the agent.
Start with early animals type of brain, very simple.

### 01/04/2023
- Setup a basic folder based on ecopy10000. Where agents controlled by a simple
neural network evolve in a 2D world trying to eat food.
- Replaced the ANN by a spiking neural network implemented in NEST
- There is no evolution for now, only a static network.
- I learned that I should not call Simulate() for each brain in the pop.

### 02/04/2023
- The code runs and the agent take action based on the output neuron that has
the highest firing rate. However, the network doesn't make much sense. I
want to take some time to understand the behaviours of the network first.
- NEST is quite slow, even for a restricted pop size.
- I did small tries with networks that include inhibitory neurons but I am
finding myself asking more questions:
    - How do I regulate the spiking rate at the circuit level ? Eg: In my
    current implementation, the output spiking rate is dependent on the number
    of neurons in the network. Should I add a permanent noise level whose
    amplitude increase and decrease to regulate the activity of the network ?
    - How do I create a realistic sensory input ? Rate base approach ? 
    - Do biological inhibitory neurons perform STDP ?
- I could continue the way it was working before, but it is similar to
a layered network and that could be done with a simple ANN. If I consider
that the layers in a deep ANN are equivalent to the micro-columns in the
neocortex, then ANN are already a good approximation of the brain.
If that is the case, then what is it that I am trying to gain from using
a spiking neural network ? I think I am curious about (1) how the brain does
what it does with sparse representations (2) how can it run on so little
energy and (3) how to implement learning and RL in a spiking neural network.
I do not have to care about (2) for now, that is purely a hardware question that
is solved by using neuromophic chips.
- I think that I should try with much simpler networks first. What is tricky
is that it is likely that the first behaviour that appeared in animals were
evolves and not learned. Learning must have come later as a way to adapt.
So I have two ways to start:
    - Use natural selection to evolve random networks with fixed weight and see
    what happens.
    - Start with a simple network and tune it by hand using survival as a metric.

I prefer the first approach. I need to generate many random architectures and
test them on a survival task.
The fixed things in the task will be:
    - The encoding of the visual inputs (rate based)
    - The encoding of the output (rate based)
The rest, the network size, architecture, weights, etc. will be random.
The network will be allowed to have from 1 to 10 networks, each having 10 to 100
neurons. The different networks will be connected to each other or not. The 
statistics of these connections can also vary (all_to_all, one_to_one,
Bernoulli, etc).

What I need to do:
- Setup the benchmark
- Setup the random network generator
- Loop over the networks and run the benchmark
- Output the scores (food eaten during a given number of simulation steps)

I implemented it. I had to add a noisy offset to regulate the global activity.
I also made the output neurons be random neurons in one of the brain regions.

I need to add a model to save and load the best network !
See https://nest-simulator.readthedocs.io/en/v3.3/auto_examples/store_restore_network.html

### 03/04/2023
- Using a simple evolutionary strategy I hope to get insight into what makes
a useful neural network without having to dig in all of the biological details of
the complex mammalian brain.
- If the brain is created using an evolutionary strategy, it will also tune the
brain to the nature of the simulation, to what is possible in this particular
simulation/brain engine.
- The final solutions can then be compared between themselves or with biological
brains to try to find patterns.
- Started to implement functions to save and load the brain.
- Run the code for 2/3 hours got some models. Including one with a score of 11.

### 05/04/2023
- Coding the loader for the models. It doesn't work yet. -> Fixed ?
- I should have a different noise generator for each brain region. The threshold
to change the mean of the generator should be in mean firing rate and not in absolute
firing rate. -> Done ?
- The weights are the same for all the connections between 2 brain regions, they
should be randomized instead -> I am not sure about that. If the different networks
represent neuron types and not brain regions, then the weights should be the same since
it is the weights between two populations. I implemented it anyway.

One thing I could do is to assign the brain_region as being inhibitory or excitatory
and then draw the weights from a different distribution depending on the type of
connection. -> Done, instead I assigned connections as being excitatory or inhibitory
with a probability of 0.75/0.25 respectively.

Refactored the code a bit
 
### 07/04/2023
- I want to make the task and the sensory inout simpler. The observation will
only be the "color" (type of object) and not the distance anymore. The distance
can be guessed anyway if the object appear in several vision rays anyway. I
removed the detection of walls for now as well. I reduced the number of vision
rays from 13 to 9.

I need to test the network on several epoch and take the average of the scores.
The agents should have hunger as an input.
Should I add plasticity ? Not for now ?

### 08/04/2023
- Critters are now tested 5 times in a row and the average score is taken.
- I now clip the hunger to 0 if it is <0 (making the task harder).
- Decrease the max number of neurons and max number of brain regions to speed
up the simulations.

I could fix the architecture of the brain and only randomize the weights. This
way, they insight I would gain would be about the neural dynamics in the neuron
population which would be easier to understand.
And in a second time, I could randomize the architecture and the weights using
these insights.

I can do the same experiment with different architecture:
- 1 excitatory only network of N neurons with recurrent connections.
- 1 excitatory and 1 inhibitory network of N neurons with recurrent
connections (Brunel 2000).
- Random architecture.

Experiment 1:
So for the first experiment, I will go with a single excitatory only network
of 1000 neurons. The delay is fixed to 1ms and the connection is pairwise
Bernoulli with a probability of 0.1. The weights are drawn at random from a
uniform distribution between 0 and 1.
The input is encoded using a rate based approach, with input of 68000Hz from
the poisson generators when input is 1 and 0Hz when input is 0.
For each step of the environment, the SNN is run for 100ms.
I will save all brains along with their scores. Once many agents have
been tested, I will analyse them all and see what patterns correlate with
success in the task at hand.

- I removed the action doing nothing and instead set that an action is "on" if
the output is >2Hz.
- Fixed a huge bug in the computation of the frequencies of the motor neurons.

- I now save the E_Ls when pickling the brains.

I think I am losing my time starting with so many neurons. I should start
with something much simpler, such as 1 network of 20 neurons bernoulli 0.5 or
something. -> Done
And I need to add an input for hunger. -> Done

I need to add plotting of the spike trains of all neurons. -> Done

One thing that confuses me is that in evolution in early animals, the "weights"
and the individual connections must have evolved and not the learning rules.
But in modern mammals it is the learning rules that evolve through evolution
and not the individual weights.
So there is one point were some part of the system became complex enough, that
it needed a dynamic control system. That must be the onset of advanced
cognitive abilities and learning rules.

I am still not happy with the encoding of the actions. I do not know how to 
threshold the output of the "motor" neurons.

I am not very happy with homeostatic regulation of excitability either.

How did neuron started in early animals ? Were they all excitatory ? Were they
even spiking ?

### 12/04/2023
- The freq was computed wrong. Now the brain activity looks better.
- Set the threshold to 1 (10Hz0 for the motor neurons.

I am still not sure about the encoding of the actions. I need to think about it
more.

I should implement an evolutionary strategy. Each population could be 200 
agents. I could keep the best 10, have them mate and mutate to get to 200
agents again.

I have doubt about how the recurrent connections themselves should evolve.
Should the plasticity rule evolve instead ?
But in early animals, the plasticity rule was not there.

I implemented the mating strategy. It needs to run.

I will need to implement the mutations as well.

### 13/04/2023
It runs until the 25th generation. It does not look like it evolved. 
I need to check for bugs in the mating and mutation.
And think about how it interacts with the homeostatic regulation of the excitability.

- I fixed some minor details in the mating and mutation.
- I added inhibitory connections (25% of all recurrent connections).
  Sensory connections are always excitatory.
- The hunger was incremented twice per cycle -> Fixed
- Increased the upper bound for the firing rate to 50Hz.

Running a new test. If it doesn't work I could try with mutation only.

They are learning so well
They even learn to not move when they are full.
Should I increase the max number of step per trial from 2000 to 10000 or +inf?
-> I switched the score, the score is now the time the agents stays alive.

- Changed the logic of the generation to have innate parallelization if
  execute the code several time (all workers write to the same directory
  of models).

ISSUE: I write to a tmp.pickle which is then read again. There is a risk that
it might be overwritten by another worker.

### 14/04/2023
- Wrote a bash script to run the code in parallel.
- Made the connections all_to_all to make the mutation and mating easier.
- Made the network 50 neurons
- Implemented proper mutation and mating.
- Added thirst and Water.
- Decreased the food/water_consumption to 0.3

### 15/04/2023
- Run the code.

### 16/04/2023
- There was a bug in the code the E_L was not properly initialized when loading
  the brain. -> Fixed
- As of now the DNA of the agent is one long vector, that is, the weights of
  all the connections. I could represent the brain as a vector and do operation
  on the vector for mutation and mating. That could make the code faster.

Once the current run is finished I need to run some analysis (including the ablation
experiment and the artificial stimuli benchmark).

- Started to implement the artificial stimuli benchmark
- Created a brain analyzer module

### 17/04/2023
- Changed the name critter(s) and blob(s) to agent(s) everywhere
- Fixed the theoretical max score in the score plot
- Created a new class for the environment related functions
- Mutate differently the weights that are 0 and the weights that are not 0 (chance of deletion)
- Mating is now much faster
- Made object wall implementable and make it so the agent dies when it hits a wall
- The retina now has 3 inputs per vision rays (BGR)
- Increased the pop_size from 60 to 100
- Made it so the agent cannot spawn on a wall or to close to the outside border.
- The motor neurons are now in red on the brain activity canvas.
- Added average number of connection per neuron to brain analyzer.
- Added distribution of the weights of the connections.
- Performed a run

### 18/04/2023
- Run went well. It starts to plateau at 30 generations.
- Analysis: Ablation experiment. Remove the recurrent connections and check the scores. -> Implemented to try
- Introduced a form of curriculum learning. I could make the environment more and more
  difficult but might make the start of the evolution impossible. Instead, a way to do curriculum
  learning is to make the environment easy but with lots of room for improvement: Added Spawner
  that spawn more food/water tokens when consumed but look like a bit like Walls.
- Reduce the initial food from 80 to 70
- Decrease food consumption when no action is taken by adding a consumption factor = (0.7 + (n_actions / 10.))
- Increased the n_test_per_agent from 3 to 5 to get more reliable estimates
- Decreased the probability of creating new connections from 0.1 to 0.05
- Launched a new run

## 20/04/2023
- The excitability is controlled based on the spike history for the last 1 second (instead of 100ms)
- Increased the threshold for action from 1 to 2 spikes / 100ms
- Increased the factor of food_consumption without action from  0.7 to 1.0 to speed up the simulation
- Made the spawner slightly less red
- Implemented statistics of the level of hunger and thirst the agent eats or drink
- Implemented statistics of the reason of death (border, hunger, thirst, trap)
- Implemented statistics of the actions taken

## 23/04/2023
- Ended the run at generation 25

I need to analyze the results in details and think about where to go from there.

## 24/04/2023
Initial analysis:
- The blob do learn to move correctly to chase the food and water (I still need to check for the spawner)
- The recurrent connections are useless. I suspect it is because of the homeostatic regulation of the excitability.

## 26/04/2023:
- Removed mating
- The intensity of the motor output is now a function of the number of spikes in the last 100ms
- Changed the upper bound for the excitability from 50Hz to 20Hz
- Decreased base_food_consumption to 0.25
- Added stpd synapses with parameters that can mutate but NOT USED FOR NOW BECAUSE OF INHIBITORY CONNECTIONS
- Decreased the max sensory frequency from 1000 to 100Hz
- Increase the max connection weight to 10

I should use 2 different brain regions for the excitatory and inhibitory neurons.
Right now the plasticity rule is applied to only excitatory connections.

## 05/05/2023:
- I am switching back to ANN in this new folder. I do not see what rate coding
  alone provide that ANN cannot do. And ANN is MUCH faster.
- Therefore, I reused my brain from ecopy with 100 hidden units and 20 memory units
- I am not sure what number to use for the clip for the action output.

## 08/05/2023:
- I need to figure out why the evolution is not working. It is like nothing
  improves after the first generations (after the convergence to the best of the
  first gen) -> I increased by a lot the mutation rate and launched a new run.
  > I think it wouldn't work in a reproductive environment because most of
  the children are in average worse than their parents since it increases the
  variance. But in a non-reproductive environment it fixed my issue.
  I need to find a trade-off if I want to use a reproductive environment.

## 10/05/2023:
- Remove the concept of generation, I now always continue evolution from
  the best 10 individuals
- I do not need to add hidden layer as I already have effectively as many
    hidden layers as I need with the recurrent connection
- Start with 100 hidden neurons and add new ones with a probability of 1%
- Start with 20 memory neurons and add new ones with a probability of 1%
- the new neurons should start with weights of 0 not to perturb the network
- Run for 13736 evaluation, max score 4391

## 07/06/2023:
- Increase the probability of adding a new neuron to 20%
- Running 1432 evaluations, score3437 -> run OLD

## 08/06/2023:
- Start from a smaller number of neuron and memory neurons (n_memory=10, 
  n_hidden=50,) and decrease the probability of adding new neurons (10%)
- Run OLD2

## 08/06/2023:
- Use only the best for evolution
- Run OLD3

## 10/06/2023:
- Use 20 best and go back to 100 hidden neurons and 20 memory neurons + 200 starting ind
- run OLD4

## 11/06/2023:
- Go back to 100 starting inv / elite 10
- Try with early exit when score < current_best_score / 5

- Try -2 -2, without smaller mutations -> bof
- Go to -0.5 +0.5, without smaller mutations ->
- Go to -1, -1 with small changes (less frequent)
