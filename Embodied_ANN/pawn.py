import secrets
import numpy
import cv2

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


class Agent(Pawn):

    def __init__(self, world_size, position=None):

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
        self.max_rotation = 0.1
        self.max_speed = 5.

        # Vision
        self.n_vision_rays = 9
        self.vision_distance = 300
        self.length_vision_vector = 3
        self.n_other_info = 2
        self.n_memory_cells = 10
        self.vision_angles = numpy.asarray([-60, -45, -30, -15, 0, 15, 30, 45, 60])
        self.vision_angles = self.vision_angles * numpy.pi / 180
        self.closest_collision_per_ray = [self.vision_distance] * self.n_vision_rays
        self.closest_object_per_ray = [None] * self.n_vision_rays

        self.brain = Brain(
            n_vision_rays=self.n_vision_rays,
            length_vision_vector=self.length_vision_vector,
            n_other_inputs=self.n_other_info,
            n_memory=20,
            n_hidden=100,
        )

        self.memory = numpy.array([0] * self.brain.n_memory)

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

    def format_observation(self, no_memory=False):

        # Concatenate the colors seen by the vision rays of the retina
        observation = []
        for i, d_angle in enumerate(self.vision_angles):
            if self.closest_object_per_ray[i]:
                observation += self.color_to_input(self.closest_object_per_ray[i].color)
            else:
                observation += [0, 0, 0]

        observation += [self.hunger / 100., self.thirst / 100.]

        if no_memory:
            observation += [0] * self.brain.n_memory
        else:
            if len(self.memory) != self.brain.n_memory:
                observation += list(self.memory) + [0]
            else:
                observation += list(self.memory)

        return numpy.array(observation)

    def do_something(self, walls, no_memory=False):

        # Get the action from the brain
        actions, memory = self.brain.evaluate(self.format_observation(no_memory))

        self.memory = memory

        # Update the movement
        amplitude_rotation = actions[0] * self.max_rotation
        self.direction += amplitude_rotation

        amplitude_mov = self.max_speed * ((actions[1] + 1) / 2)
        self.position = self.position + amplitude_mov * numpy.asarray([numpy.cos(self.direction), numpy.sin(self.direction)])

        self.clip_direction()
        # self.clip_position()

        consumption_factor = (1. + (amplitude_mov / 10.))
        self.hunger += consumption_factor * self.base_food_consumption
        self.thirst += consumption_factor * self.base_food_consumption

        if self.hunger >= 100:
            self.alive = False
            self.brain.cause_of_death.append("hunger")
            return None
        if self.thirst >= 100:
            self.alive = False
            self.brain.cause_of_death.append("thirst")
            return None

        # Check if the agent is colliding with a wall/trap
        if self.position[0] < 0 or self.world_size < self.position[0] or self.position[1] < 0 or self.world_size < self.position[1]:
            self.alive = False
            self.brain.cause_of_death.append("border")
            return None
        for wall in walls:
            if (wall.position[0] - self.position[0]) ** 2 + (wall.position[1] - self.position[1]) ** 2 < wall.radius ** 2:
                self.alive = False
                self.brain.cause_of_death.append("trap")
                return None

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
                return self.closest_object_per_ray[r]

        return None
