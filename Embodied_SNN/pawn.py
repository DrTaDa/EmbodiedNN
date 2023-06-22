import secrets
import numpy
import cv2
from numba import jit

from brain import Brain


class Pawn:

    def __init__(self, position, direction, world_size, radius=0, color=(254, 254, 254)):

        self.radius = radius
        self.color = color

        self.position = position
        self.direction = direction

        self.world_size = world_size
        self.name = secrets.token_urlsafe(10)

        self.clip_position()
        self.clip_direction()

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    def clip_position(self):
        self.position = numpy.clip(self.position, 0, self.world_size)

    def clip_direction(self):
        if self.direction >= 2. * numpy.pi:
            self.direction -= 2. * numpy.pi
        elif self.direction < 0:
            self.direction += 2. * numpy.pi

    def draw(self, canvas):
        pass

    def do_something(self):
        pass



class Wall(Pawn):

    def __init__(self, world_size, position=None):

        if position is None:
            position = numpy.random.random(2) * world_size

        Pawn.__init__(
            self,
            position=position,
            direction=0.,
            world_size=world_size,
            radius=12,
            color=(0, 0, 255)
        )

    def draw(self, canvas):
        cv2.circle(canvas, (int(self.x), int(self.y)), self.radius, self.color, -1)

    def do_something(self):
        pass


class Food(Pawn):

    def __init__(self, world_size, position=None):

        if position is None:
            position = numpy.random.random(2) * world_size

        Pawn.__init__(
            self,
            position=position,
            direction=0.,
            world_size=world_size,
            radius=12,
            color=(0, numpy.random.uniform(150, 255), 0)
        )

        self.size = 50

    def draw(self, canvas):
        cv2.circle(canvas, (int(self.x), int(self.y)), self.radius, self.color, -1)

    def do_something(self):
        pass


class Water(Pawn):

    def __init__(self, world_size, position=None):

        if position is None:
            position = numpy.random.random(2) * world_size

        Pawn.__init__(
            self,
            position=position,
            direction=0.,
            world_size=world_size,
            radius=12,
            color=(numpy.random.uniform(150, 255), 0, 0)
        )

        self.size = 50

    def draw(self, canvas):
        cv2.circle(canvas, (int(self.x), int(self.y)), self.radius, self.color, -1)

    def do_something(self):
        pass


class Spawner(Pawn):

    def __init__(self, world_size, position=None):

        if position is None:
            position = numpy.random.random(2) * world_size

        Pawn.__init__(
            self,
            position=position,
            direction=0.,
            world_size=world_size,
            radius=12,
            color=(numpy.random.randint(220, 255), 0, numpy.random.randint(210, 245))
        )

        self.size = 50

    def draw(self, canvas):
        cv2.circle(canvas, (int(self.x), int(self.y)), self.radius, self.color, -1)

    def do_something(self):
        pass


#@jit(nopython=True, fastmath=True)
def jit_move_forward_clip(position, speed, direction, world_size):

    new_position = numpy.clip(
        position + speed * numpy.asarray([numpy.cos(direction), numpy.sin(direction)]),
        0,
        world_size
    )

    return new_position, numpy.abs(speed / 4.0) ** 2.5


class Agent(Pawn):

    def __init__(self, world_size, position=None, from_file=None):

        if position is None:
            position = numpy.random.random(2) * world_size

        Pawn.__init__(
            self,
            position=position,
            direction=numpy.random.random() * 2. * numpy.pi,
            world_size=world_size,
            radius=12,
            color=(50, 50, 50)
        )

        self.size = 50

        # Health
        self.hunger = 0
        self.thirst = 0
        self.base_food_consumption = 0.25
        self.alive = True

        # Movement
        self.n_actions = 3
        self.max_rotation = 0.1
        self.max_speed = 5.

        # Vision
        self.n_vision_rays = 9
        self.vision_distance = 300
        self.length_vision_vector = 3
        self.n_other_info = 2
        self.vision_angles = numpy.asarray([-60, -45, -30, -15, 0, 15, 30, 45, 60])
        self.vision_angles = self.vision_angles * numpy.pi / 180
        self.closest_collision_per_ray = [self.vision_distance] * self.n_vision_rays
        self.closest_object_per_ray = [None] * self.n_vision_rays

        # Brain
        self.brain = Brain(
            (self.n_vision_rays * self.length_vision_vector) + self.n_other_info,
            self.n_actions,
            from_file=from_file
        )

    def reset_position(self):
        self.position = numpy.random.randint(self.radius, self.world_size - self.radius, 2)
        self.clip_position()

    def reset_direction(self):
        self.direction = numpy.random.random() * 2. * numpy.pi
        self.clip_direction()

    def reset(self):
        self.reset_direction()
        self.reset_position()
        self.hunger = 0
        self.thirst = 0
        self.alive = True

    def draw(self, canvas, draw_vision=False):

        start_point = (int(self.x), int(self.y))
        movement_direction = [numpy.cos(self.direction), numpy.sin(self.direction)]
        end_point = (
            int(self.x + ((self.radius + 3) * movement_direction[0])),
            int(self.y + ((self.radius + 3) * movement_direction[1]))
        )

        cv2.line(canvas, start_point, end_point, self.color, 9)
        cv2.circle(canvas, start_point, int(self.radius), self.color, -1)

        if draw_vision:
            for i, d_angle in enumerate(self.vision_angles):
                angle_ray = self.direction + d_angle
                ray_direction = [numpy.cos(angle_ray), numpy.sin(angle_ray)]

                length = self.closest_collision_per_ray[i]
                if self.closest_collision_per_ray[i] > self.vision_distance:
                    length = self.vision_distance

                end_point = (
                    int(self.x + (length * ray_direction[0])),
                    int(self.y + (length * ray_direction[1]))
                )

                if self.closest_object_per_ray[i]:
                    if isinstance(self.closest_object_per_ray[i], Wall):
                        cv2.line(canvas, start_point, end_point, self.closest_object_per_ray[i].color, 1)
                    else:
                        cv2.line(canvas, start_point, end_point, self.closest_object_per_ray[i].color, 1)
                        end_point2 = (int(self.closest_object_per_ray[i].x),
                                      int(self.closest_object_per_ray[i].y))
                        cv2.line(canvas, end_point, end_point2, self.closest_object_per_ray[i].color, 1)
                else:
                    cv2.line(canvas, start_point, end_point, (255, 255, 255), 1)

    def color_to_input(self, color):
        return [color[0] / 255., color[1] / 255., color[2] / 255.]

    def receive_inputs(self):

        # Concatenate the colors seen by the vision rays of the retina
        observation = []
        for i, d_angle in enumerate(self.vision_angles):
            if self.closest_object_per_ray[i]:
                observation += self.color_to_input(self.closest_object_per_ray[i].color)
            else:
                observation += [0, 0, 0]

        observation += [self.hunger / 100., self.thirst / 100.]

        self.brain.prepare_simulation(observation)

    def do_something(self, walls):

        # Get the action from the brain
        actions, spike_trains = self.brain.get_action()

        # Update the movement
        self.brain.actions_history.append([])

        amplitude_left = actions[0] / 5
        if amplitude_left > 1:
            amplitude_left = 1
        self.direction -= self.max_rotation * amplitude_left
        self.brain.actions_history[-1].append(amplitude_left)

        amplitude_right = actions[1] / 5
        if amplitude_right > 1:
            amplitude_right = 1
        self.direction += self.max_rotation * amplitude_right
        self.brain.actions_history[-1].append(amplitude_right)

        amplitude_forward = actions[2] / 5
        if amplitude_forward > 1:
            amplitude_forward = 1
        self.position = self.position + self.max_speed * amplitude_forward * numpy.asarray([numpy.cos(self.direction), numpy.sin(self.direction)])

        self.clip_direction()
        # self.clip_position()

        total_amplitude = amplitude_left + amplitude_right + amplitude_forward
        consumption_factor = (1. + (total_amplitude / 10.))
        self.hunger += consumption_factor * self.base_food_consumption
        self.thirst += consumption_factor * self.base_food_consumption

        if self.hunger >= 100:
            self.alive = False
            self.brain.cause_of_death.append("hunger")
            return None, None
        if self.thirst >= 100:
            self.alive = False
            self.brain.cause_of_death.append("thirst")
            return None, None

        # Check if the agent is colliding with a wall/trap
        if self.position[0] < 0 or self.world_size < self.position[0] or self.position[1] < 0 or self.world_size < self.position[1]:
            self.alive = False
            self.brain.cause_of_death.append("border")
            return None, spike_trains
        for wall in walls:
            if (wall.position[0] - self.position[0]) ** 2 + (wall.position[1] - self.position[1]) ** 2 < wall.radius ** 2:
                self.alive = False
                self.brain.cause_of_death.append("trap")
                return None, spike_trains

        for r in numpy.argsort(self.closest_collision_per_ray):
            if self.closest_object_per_ray[r] is not None and self.closest_collision_per_ray[r] < 12:
                if isinstance(self.closest_object_per_ray[r], Food):
                    self.brain.hunger_when_eating.append(self.hunger)
                    self.hunger -=  self.closest_object_per_ray[r].size
                    if self.hunger < 0:
                        self.hunger = 0
                if isinstance(self.closest_object_per_ray[r], Water):
                    self.brain.thirst_when_drinking.append(self.thirst)
                    self.thirst -=  self.closest_object_per_ray[r].size
                    if self.thirst < 0:
                        self.thirst = 0
                return self.closest_object_per_ray[r], spike_trains

        return None, spike_trains
