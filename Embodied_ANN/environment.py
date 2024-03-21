import cv2
import numpy
from scipy.spatial.distance import cdist
from numba import jit
from pawn import Agent, Food, Wall, Water, Spawner


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


@jit(nopython=True, fastmath=True)
def jit_find_closest_index(distances, distances_to_ray_end, rays_dx, rays_dy,
                           diff_distances,
                           out_ray):
    min_distance_to_end = numpy.minimum(distances, distances_to_ray_end)
    distance_to_seg = numpy.abs(
        (rays_dy * diff_distances[:, 0]) - (rays_dx * diff_distances[:, 1])) / 300
    shortest_distances_to_ray = numpy.where(out_ray, min_distance_to_end, distance_to_seg)
    return (shortest_distances_to_ray < 12).nonzero()[0]


def check_vision_of_walls(agents, i, r, ray_end_points_x, ray_end_points_y, world_size,
                          ray_end):

    if agents[i].closest_object_per_ray[r] is None:

        closest_collision = 1000
        if ray_end_points_x[i][r] < 0:
            line1 = [agents[i].position, ray_end[i][r]]
            line2 = [[0, 0], [0, world_size]]
            point_collision = line_intersection(line1, line2)
            distance_collision = ((agents[i].position[0] - point_collision[0]) ** 2 + (
                    agents[i].position[1] - point_collision[1]) ** 2) ** 0.5
            if distance_collision <= closest_collision:
                closest_collision = distance_collision
        if ray_end_points_x[i][r] > world_size:
            line1 = [agents[i].position, ray_end[i][r]]
            line2 = [[world_size, 0], [world_size, world_size]]
            point_collision = line_intersection(line1, line2)
            distance_collision = ((agents[i].position[0] - point_collision[0]) ** 2 + (
                    agents[i].position[1] - point_collision[1]) ** 2) ** 0.5
            if distance_collision <= closest_collision:
                closest_collision = distance_collision
        if ray_end_points_y[i][r] < 0:
            line1 = [agents[i].position, ray_end[i][r]]
            line2 = [[0, 0], [world_size, 0]]
            point_collision = line_intersection(line1, line2)
            distance_collision = ((agents[i].position[0] - point_collision[0]) ** 2 + (
                    agents[i].position[1] - point_collision[1]) ** 2) ** 0.5
            if distance_collision <= closest_collision:
                closest_collision = distance_collision
        if ray_end_points_y[i][r] > world_size:
            line1 = [agents[i].position, ray_end[i][r]]
            line2 = [[0, world_size], [world_size, world_size]]
            point_collision = line_intersection(line1, line2)
            distance_collision = ((agents[i].position[0] - point_collision[0]) ** 2 + (
                    agents[i].position[1] - point_collision[1]) ** 2) ** 0.5
            if distance_collision <= closest_collision:
                closest_collision = distance_collision

        if closest_collision <= 300:
            agents[i].closest_collision_per_ray[r] = closest_collision
            agents[i].closest_object_per_ray[r] = Wall(1, [0, 0])


def compute_vision(agents, entities, vision_distance, world_size):
    """Batch processing of ray casting and vision.

    WARNING: HARD-CODED LENGTH AND DISTANCES"""

    entities_ = agents + entities
    positions = numpy.asarray([e.position for e in entities_])
    angle_rays = agents[0].vision_angles[:, numpy.newaxis] + numpy.asarray(
        [e.direction for e in agents])

    distances = cdist(positions[:len(agents)], positions)

    rays_dx = vision_distance * numpy.cos(angle_rays.T)
    rays_dy = vision_distance * numpy.sin(angle_rays.T)
    ray_end_offset = numpy.stack([rays_dx, rays_dy], axis=-1)
    ray_end_points_x = positions[:len(agents), 0, numpy.newaxis] + rays_dx
    ray_end_points_y = positions[:len(agents), 1, numpy.newaxis] + rays_dy
    ray_end = numpy.stack([ray_end_points_x, ray_end_points_y], axis=-1)

    for i, b in enumerate(agents):

        diff_distances = positions - b.position
        dets = numpy.dot(ray_end_offset[i], diff_distances.T)
        out_ray = numpy.logical_or(dets < 0, dets > 90000)
        distances_to_ray_end = cdist(ray_end[i], positions)

        agents[i].closest_collision_per_ray = [100000] * b.n_vision_rays
        agents[i].closest_object_per_ray = [None] * b.n_vision_rays

        for r in range(b.n_vision_rays):

            close_indexes = jit_find_closest_index(
                distances[i], distances_to_ray_end[r], rays_dx[i, r], rays_dy[i, r],
                diff_distances,
                out_ray[r]
            )

            if len(close_indexes) > 1:
                j = numpy.argsort(distances[i, close_indexes])[1]
                agents[i].closest_collision_per_ray[r] = distances[i, close_indexes[j]]
                agents[i].closest_object_per_ray[r] = entities_[close_indexes[j]]

            # Check for collisions with walls
            if agents[i].closest_object_per_ray[r] is None:
                check_vision_of_walls(agents, i, r, ray_end_points_x, ray_end_points_y,
                                      world_size, ray_end)


class Environment:

    def __init__(self, world_size, vision_distance, n_food=70, n_walls=5, n_spawners=2, token_per_spawner=20):

        self.world_size = world_size
        self.vision_distance = vision_distance
        self.n_food = n_food
        self.n_walls = n_walls
        self.n_spawners = n_spawners
        self.token_per_spawner = token_per_spawner

    def draw_pawns(self, canvas, entities, agents):

        # Draw the outside walls;
        cv2.rectangle(canvas, (0, 0), (self.world_size, self.world_size), (0, 0, 255), 2)

        for e in entities:
            e.draw(canvas)
        for c in agents:
            c.draw(canvas, draw_vision=True)

    def draw_general_stats(self, canvas, score, agent):
        # Display some stats:
        cv2.putText(canvas, 'Score: ' + str(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(canvas, 'Hunger: ' + str(int(agent.hunger)), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'Thirst: ' + str(int(agent.thirst)), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'Name: ' + str(agent.name), (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

    def draw_memory(self, canvas, agent):
        for i, m in enumerate(agent.memory):
            color = (int(m * 255), int(m * 255), int(m * 255))
            cv2.circle(canvas, ((1 + i) * 22, 150), 10, color, -1)

    def draw_and_display(self, entities, agent, score):

        canvas = numpy.full((self.world_size, self.world_size, 3), 100, numpy.uint8)
        self.draw_pawns(canvas, entities, [agent])
        self.draw_general_stats(canvas, score, agent)
        self.draw_memory(canvas, agent)

        cv2.imshow('SNN Ecopy 10000', canvas)
        cv2.waitKey(1)

    def spawn_entities_non_overlapping(self, walls, class_, n):
        entities = []
        while len(entities) < n:
            e = class_(self.world_size)
            if numpy.min(cdist([e.position], [w.position for w in walls])) > walls[0].radius * 2:
                entities.append(e)
        return entities

    def spawn_food(self, walls, n_food):
        return self.spawn_entities_non_overlapping(walls, Food, n_food)

    def spawn_water(self, walls, n_water):
        return self.spawn_entities_non_overlapping(walls, Water, n_water)

    def spawn_spawners(self, walls, n_spawners):
        return self.spawn_entities_non_overlapping(walls, Spawner, n_spawners)

    def init_entities(self):

        walls = [Wall(self.world_size) for _ in range(self.n_walls)]

        foods = self.spawn_food(walls, self.n_food)
        waters = self.spawn_water(walls, self.n_food)
        spawners = self.spawn_spawners(walls, self.n_spawners)

        return walls, foods + waters + spawners

    def test_agent(self, agent, display=True, no_memory=False, no_hunger_signal=False):

        walls, entities = self.init_entities()
        entities += walls

        while numpy.min(cdist([agent.position], [w.position for w in walls])) < walls[0].radius * 2:
            agent.reset_position()

        score = 0
        while agent.alive:

            # Compute vision
            compute_vision([agent], entities, self.vision_distance, self.world_size)

            # Display the environment
            if display:
                self.draw_and_display(entities, agent, score)

            # Actions
            target = agent.do_something(walls, no_memory, no_hunger_signal)
            if target is not None:
                entities = [e for e in entities if e.name != target.name]
                if isinstance(target, Spawner):
                    entities += self.spawn_food(walls, self.token_per_spawner)
                    entities += self.spawn_water(walls, self.token_per_spawner)

            score += 1

        return score

    def test_agent_memory(self, agent, display=True, no_memory=False):

        memories = []
        positions = []
        directions = []

        walls, entities = self.init_entities()
        entities += walls

        while numpy.min(cdist([agent.position], [w.position for w in walls])) < walls[0].radius * 2:
            agent.reset_position()

        score = 0
        while agent.alive:

            positions.append(agent.position)
            directions.append(agent.direction)

            # Compute vision
            compute_vision([agent], entities, self.vision_distance, self.world_size)

            # Display the environment
            if display:
                self.draw_and_display(entities, agent, score)

            # Actions
            target = agent.do_something(walls, no_memory)
            memories.append(agent.memory)

            if target is not None:
                entities = [e for e in entities if e.name != target.name]
                if isinstance(target, Spawner):
                    entities += self.spawn_food(walls, self.token_per_spawner)
                    entities += self.spawn_water(walls, self.token_per_spawner)

            score += 1

        return score, memories, positions, directions
