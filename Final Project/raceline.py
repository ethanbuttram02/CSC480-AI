import pygame
import numpy as np
import math
from random import randrange
import random
from typing import List, Tuple
import sys

pygame.init()  # initialize pygame

# constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
FPS = 500
POPULATION_SIZE = 50
MUTATION_RATE = 0.15  # increased base mutation rate
CROSSOVER_RATE = 0.7
GENERATIONS = 100
SPEED = 10.0  # constant speed for all cars

# dynamic mutation parameters
MIN_MUTATION_RATE = 0.1
MAX_MUTATION_RATE = 0.4
STAGNATION_THRESHOLD = 10  # generations without improvement

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# x and y offsets
GLOBAL_X_OFFSET = 200
GLOBAL_Y_OFFSET = 0


class Track:
    def __init__(self):
        def offset_points(points):
            return [(x + GLOBAL_X_OFFSET, y + GLOBAL_Y_OFFSET) for x, y in points]

        self.outer_points = offset_points([
            (190, 140), (350, 110), (500, 120), (580, 140),
            (920, 140), (960, 180), (980, 230), (990, 280),
            (1000, 340), (1020, 420), (1020, 500), (1000, 580),
            (930, 640), (840, 690), (720, 750), (620, 760), (480, 770),
            (340, 750), (260, 720),
            (135, 640), (100, 580), (110, 530), (190, 500),
            (150, 450), (70, 450),
            (50, 380), (50, 320), (80, 260), (125, 205)
        ])

        self.inner_points = offset_points([
            (250, 200), (350, 200), (420, 210), (480, 220), (540, 225),
            (600, 225), (660, 215), (720, 210),
            (860, 210), (880, 210), (900, 220), (910, 270), (930, 320),
            (930, 400), (930, 470), (925, 530), (920, 560), (760, 660),
            (660, 680), (500, 700), (400, 690), (350, 670), (265, 630),
            (260, 620), (240, 570),
            (315, 520), (200, 380), (140, 380),
            (140, 360), (160, 300), (200, 230)
        ])

        self.waypoints = offset_points([
            (440, 170), (710, 150), (870, 190), (940, 260), (1000, 440),
            (990, 560), (880, 630), (700, 710), (550, 750), (370, 730),
            (240, 640), (200, 560), (210, 470), (110, 400), (130, 270),
            (230, 170), (390, 165)
        ])

        self.start_pos = (400 + GLOBAL_X_OFFSET, 170 + GLOBAL_Y_OFFSET)
        self.start_angle = 0
        self.finish_line = offset_points([(400, 110), (400, 210)])
        self.min_distance_before_finish = 150

    def draw(self, screen):
        pygame.draw.lines(screen, BLACK, True, self.outer_points, 3)
        pygame.draw.lines(screen, BLACK, True, self.inner_points, 3)
        pygame.draw.line(screen, RED, self.finish_line[0], self.finish_line[1], 3)
        for i, waypoint in enumerate(self.waypoints):
            pygame.draw.circle(screen, GRAY, waypoint, 6)
            font = pygame.font.Font(None, 20)
            text = font.render(str(i), True, BLACK)
            screen.blit(text, (waypoint[0] + 8, waypoint[1] - 8))

    def is_on_track(self, pos):
        if not hasattr(self, 'collision_mask'):
            mask_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            mask_surface.fill((0, 0, 0, 0))
            pygame.draw.polygon(mask_surface, (255, 255, 255, 255), self.outer_points)
            pygame.draw.polygon(mask_surface, (0, 0, 0, 0), self.inner_points)
            self.collision_mask = pygame.mask.from_surface(mask_surface)

        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            return self.collision_mask.get_at((x, y)) == 1
        return False

    def crossed_finish_line(self, old_pos, new_pos, distance_traveled):
        if distance_traveled < self.min_distance_before_finish:
            return False
        x1, y1 = old_pos
        x2, y2 = new_pos
        x3, y3 = self.finish_line[0]
        x4, y4 = self.finish_line[1]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return False
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        return 0 <= t <= 1 and 0 <= u <= 1

    def get_progress(self, pos, current_checkpoint=0):
        target = self.waypoints[current_checkpoint % len(self.waypoints)]
        dist = math.hypot(pos[0] - target[0], pos[1] - target[1])
        checkpoint_radius = 60  # increased from 60
        if dist < checkpoint_radius:
            return current_checkpoint + 1, True
        max_distance = 500  # increased from 200
        progress = max(0, (max_distance - dist) / max_distance)
        return current_checkpoint + progress, False


class Car:
    def __init__(self, genome=None):
        self.pos = list(track.start_pos)
        self.angle = track.start_angle
        self.genome = genome if genome else [random.uniform(-0.1, 0.1) for _ in range(400)]
        self.fitness = 0
        self.finished = False
        self.crashed = False
        self.time_alive = 0
        self.distance_traveled = 0
        self.path = [tuple(self.pos)]
        self.gene_index = 0
        self.max_progress = 0.0
        self.current_checkpoint = 0
        self.checkpoints_reached = []
        self.backwards_time = 0
        self.stuck_time = 0
        self.last_position = tuple(self.pos)
        self.last_progress_update = 0

    def is_car_on_track(self, pos, radius=1):
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            offset = (pos[0] + math.cos(rad) * radius, pos[1] + math.sin(rad) * radius)
            if not track.is_on_track(offset):
                return False
        return True

    def update(self):
        if self.time_alive > 60 and self.max_progress < 0.5:  # early termination for non-progressing cars
            self.crashed = True
            self.fitness = self.calculate_fitness()
            return
        if self.crashed or self.finished:
            return

        steering = self.genome[self.gene_index] if self.gene_index < len(self.genome) else self.genome[-1]
        self.gene_index += 1
        steering = max(-10, min(10, steering))  # allow sharper turns
        self.angle += steering

        old_pos = tuple(self.pos)
        dx = math.cos(self.angle) * SPEED
        dy = math.sin(self.angle) * SPEED
        self.pos[0] += dx
        self.pos[1] += dy
        self.distance_traveled += SPEED

        if not self.is_car_on_track(self.pos):
            self.crashed = True
            self.fitness = self.calculate_fitness()
            return

        progress_value, reached_checkpoint = track.get_progress(self.pos, self.current_checkpoint)
        if reached_checkpoint:
            self.current_checkpoint += 1
            self.checkpoints_reached.append(self.current_checkpoint)
            self.max_progress = self.current_checkpoint
            self.backwards_time = 0
            self.last_progress_update = self.time_alive
        else:
            new_progress = progress_value
            if new_progress > self.max_progress:
                self.max_progress = new_progress
                self.backwards_time = 0
                self.last_progress_update = self.time_alive
            elif self.time_alive - self.last_progress_update > 120:
                self.backwards_time += 1

        pos_change = math.hypot(self.pos[0] - self.last_position[0], self.pos[1] - self.last_position[1])
        self.stuck_time = self.stuck_time + 1 if pos_change < 1.0 else 0
        self.last_position = tuple(self.pos)

        if self.backwards_time > 100 or self.stuck_time > 120:
            self.crashed = True
            self.fitness = self.calculate_fitness()
            return

        if track.crossed_finish_line(old_pos, self.pos, self.distance_traveled):
            self.finished = True
            self.fitness = self.calculate_fitness()
            return

        self.path.append(tuple(self.pos))
        self.time_alive += 1

        if self.time_alive > 2000:
            self.crashed = True
            self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        if not self.finished and self.time_alive < 30:  # kill early crashers with low fitness
            return 1
        if self.finished:
            return 75000 + (4000 - min(4000, self.time_alive)) + len(self.checkpoints_reached) * 1500
        else:
            checkpoint_fitness = len(self.checkpoints_reached) * 4000
            partial_progress = (self.max_progress % 1) * 1000
            time_fitness = min(self.time_alive * 2, 1000)
            if hasattr(track, 'waypoints') and len(track.waypoints) > 0:
                target_waypoint = track.waypoints[self.current_checkpoint % len(track.waypoints)]
                distance_to_target = math.hypot(self.pos[0] - target_waypoint[0], self.pos[1] - target_waypoint[1])
                proximity_bonus = max(0, 100 - distance_to_target / 2)
            else:
                proximity_bonus = 0
            return max(1, checkpoint_fitness + partial_progress + time_fitness + proximity_bonus)

    def draw(self, screen, color=BLUE, size=3):
        if len(self.path) > 1:
            pygame.draw.lines(screen, color, False, self.path, 2)
        if not self.crashed or len(self.path) > 1:
            pygame.draw.circle(screen, color, (int(self.pos[0]), int(self.pos[1])), size)
        if not self.crashed:
            end_x = self.pos[0] + math.cos(self.angle) * 10
            end_y = self.pos[1] + math.sin(self.angle) * 10
            pygame.draw.line(screen, color, self.pos, (end_x, end_y), 2)


class GeneticAlgorithm:
    def __init__(self):
        self.population = [Car() for _ in range(POPULATION_SIZE)]
        self.generation = 0
        self.best_car = None
        self.best_fitness = 0
        self.last_improvement = 0
        self.stagnation_count = 0
        self.current_mutation_rate = MUTATION_RATE

    def evaluate_population(self):  # calculate fitness for current population after simulation
        for car in self.population:  # calculate fitness for all cars
            if car.fitness == 0:
                car.fitness = car.calculate_fitness()

        current_best = max(self.population, key=lambda x: x.fitness)  # update best car and check for stagnation
        if current_best.fitness > self.best_fitness:
            self.best_fitness = current_best.fitness
            self.best_car = current_best
            self.last_improvement = self.generation
            self.stagnation_count = 0
        elif current_best.fitness == self.best_fitness and current_best is not self.best_car:
            self.best_car = current_best  # new car tied best
            self.last_improvement = self.generation
            self.stagnation_count += 1
        else:
            self.stagnation_count += 1

        if self.stagnation_count > STAGNATION_THRESHOLD:  # adjust mutation rate based on stagnation
            self.stagnation_count = 0
            self.current_mutation_rate = min(MAX_MUTATION_RATE, self.current_mutation_rate * 1.3)
        else:
            self.current_mutation_rate = max(MIN_MUTATION_RATE, self.current_mutation_rate * 0.95)

    def selection(self, population):  # tournament selection with fitness-based selection pressure
        tournament_size = 5
        selected = []

        population.sort(key=lambda x: x.fitness, reverse=True)  # sort population by fitness for better selection pressure

        for _ in range(len(population)):
            if random.random() < 0.3:  # 30% chance to select from top performers
                top_count = max(1, len(population) // 5)  # select from top 20% of population
                selected.append(random.choice(population[:top_count]))
            else:
                tournament = random.sample(population, min(tournament_size, len(population)))  # tournament selection
                winner = max(tournament, key=lambda x: x.fitness)
                selected.append(winner)

        return selected

    def crossover(self, parent1, parent2):  # multi-point crossover for better gene mixing
        if random.random() > CROSSOVER_RATE:
            return Car(parent1.genome[:]), Car(parent2.genome[:])

        genome_length = len(parent1.genome)  # use 2-3 crossover points for better mixing
        num_points = random.randint(2, 3)
        crossover_points = sorted(random.sample(range(1, genome_length), num_points))

        child1_genome = []
        child2_genome = []

        use_parent1 = True
        last_point = 0

        for point in crossover_points + [genome_length]:
            if use_parent1:
                child1_genome.extend(parent1.genome[last_point:point])
                child2_genome.extend(parent2.genome[last_point:point])
            else:
                child1_genome.extend(parent2.genome[last_point:point])
                child2_genome.extend(parent1.genome[last_point:point])

            use_parent1 = not use_parent1
            last_point = point

        return Car(child1_genome), Car(child2_genome)

    def mutate(self, car):  # gaussian mutation with adaptive rate and targeted mutations
        for i in range(len(car.genome)):
            if random.random() < self.current_mutation_rate:
                mutation_strength = 0.05 if self.stagnation_count <= STAGNATION_THRESHOLD else 0.1  # use smaller, more controlled mutations
                car.genome[i] += random.gauss(0, mutation_strength)
                car.genome[i] = max(-0.3, min(0.3, car.genome[i]))  # clamp values

        if self.stagnation_count >= STAGNATION_THRESHOLD and random.random() < 0.15:  # occasionally apply larger mutations to break out of local optima
            start_idx = random.randint(0, len(car.genome) - 20)  # replace a small segment with new random values
            end_idx = start_idx + random.randint(5, 20)
            for i in range(start_idx, min(end_idx, len(car.genome))):
                car.genome[i] = random.uniform(-0.2, 0.2)

    def evolve(self):  # create next generation
        selected = self.selection(self.population)  # selection

        new_population = []  # create new population

        elite_count = max(2, POPULATION_SIZE // 10)  # keep best individuals (elitism) - keep top 10%
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        for i in range(elite_count):
            new_population.append(Car(sorted_population[i].genome[:]))

        diversity_count = max(1, POPULATION_SIZE // 50) if self.stagnation_count <= STAGNATION_THRESHOLD else max(2, POPULATION_SIZE // 20)  # add some random individuals to maintain diversity (fewer when not stagnating)
        for _ in range(diversity_count):
            new_population.append(Car())  # completely random genome

        while len(new_population) < POPULATION_SIZE:  # generate rest of population through crossover and mutation
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child1, child2 = self.crossover(parent1, parent2)

            self.mutate(child1)
            self.mutate(child2)

            new_population.extend([child1, child2])

        self.population = new_population[:POPULATION_SIZE]
        self.generation += 1


def interpolate_path(path, resolution=3):
    if len(path) < 2:
        return path
    result = []
    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, int(dist / resolution))
        for s in range(steps):
            x = p1[0] + (p2[0] - p1[0]) * (s / steps)
            y = p1[1] + (p2[1] - p1[1]) * (s / steps)
            result.append((x, y))
    result.append(path[-1])
    return result


def interpolate_line(p1, p2, resolution=3):  # returns list of evenly spaced points from p1 to p2
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    steps = max(1, int(dist / resolution))
    return [
        (p1[0] + (p2[0] - p1[0]) * i / steps,
         p1[1] + (p2[1] - p1[1]) * i / steps)
        for i in range(steps + 1)
    ]


def main():
    global track
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("GA Optimal \"Race\" Line ")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    track = Track()
    ga = GeneticAlgorithm()

    simulating = False
    simulation_speed = 1

    auto_sim_mode = False  # automation control
    auto_sim_count = 1000
    car_finished = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not simulating:
                        auto_sim_count = 1000
                        simulating = True
                        if car_finished:
                            simulating = False
                            auto_sim_mode = False
                            car_finished = False
                        else:
                            auto_sim_mode = True
                        for car in ga.population:
                            car.__init__(car.genome[:])
                elif event.key == pygame.K_r:
                    ga = GeneticAlgorithm()
                    car_finished = False
                    auto_sim_mode = False
                    auto_sim_count = 0
                    simulating = False
                elif event.key == pygame.K_UP:
                    simulation_speed = min(100, simulation_speed + 10)
                elif event.key == pygame.K_DOWN:
                    simulation_speed = max(1, simulation_speed - 1)

        if simulating:
            for _ in range(simulation_speed):
                active_cars = [car for car in ga.population if not car.crashed and not car.finished]

                if not active_cars:
                    ga.evaluate_population()
                    ga.evolve()

                    if any(car.finished for car in ga.population):
                        car_finished = True
                        simulating = False
                        auto_sim_mode = False
                    elif auto_sim_mode:
                        auto_sim_count -= 1
                        if auto_sim_count > 0:
                            for car in ga.population:
                                car.__init__(car.genome[:])
                        else:
                            simulating = False
                            auto_sim_mode = False
                    else:
                        simulating = False
                    break

                for car in active_cars:
                    car.update()

        screen.fill(WHITE)
        track.draw(screen)

        for car in ga.population:
            car.draw(screen, GREEN if car.finished else RED if car.crashed else BLUE)

        if ga.best_car and ga.best_car.path:
            pygame.draw.lines(screen, PURPLE, False, ga.best_car.path, 4)
            pygame.draw.circle(screen, PURPLE, (int(ga.best_car.path[-1][0]), int(ga.best_car.path[-1][1])), 6)

        screen.blit(font.render(f"Gen: {ga.generation}", True, BLACK), (10, 10))  # ui
        screen.blit(font.render(f"Best Fitness: {ga.best_fitness:.1f}", True, BLACK), (10, 50))
        if ga.best_car:
            screen.blit(font.render(f"Checkpoints: {len(ga.best_car.checkpoints_reached)}/{len(track.waypoints)}", True, BLACK), (10, 90))
        screen.blit(font.render(f"Mutation Rate: {ga.current_mutation_rate:.3f}", True, BLACK), (10, 130))
        screen.blit(font.render(f"Stagnation: {ga.stagnation_count}/{STAGNATION_THRESHOLD}", True, BLACK), (10, 170))
        screen.blit(font.render(f"Speed: {simulation_speed}x", True, BLACK), (10, 210))
        screen.blit(font.render(f"Active Cars: {len([c for c in ga.population if not c.crashed and not c.finished])}", True, BLACK), (10, 250))

        if simulating:
            screen.blit(font.render("SIMULATING...", True, GREEN), (SCREEN_WIDTH - 300, 10))
        else:
            screen.blit(font.render("PRESS SPACE TO START", True, RED), (SCREEN_WIDTH - 300, 10))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()