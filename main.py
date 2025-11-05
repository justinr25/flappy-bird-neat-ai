import sys
import time
import random
import json
import os
import math

import pygame
import neat

import visualize
from player import Player
from obstacle import Obstacle
from reset_button import ResetButton
from utils import display_text
from utils import calculate_distance

class Simulation:
    def __init__(self):
        # setup pygame
        pygame.init()
        pygame.display.set_caption('Flappy Bird NEAT AI')
        self.monitor_size = [pygame.display.Info().current_w, pygame.display.Info().current_h]
        self.is_fullscreen = False
        self.screen_size = 1280, 720
        self.max_fps = 60
        self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.last_time = time.perf_counter()
        self.screen_bg_color = (255, 255, 255)
        self.is_simulation_started = False
        self.score = 0
        self.generation_num = 0

        # load NEAT-Python configuration file
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward.txt')

        # NEAT setup
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)

    def simulation_setup(self, genomes):
        # NEAT settup
        self.nets = [] # neural network of each player
        self.ge = [] # genome of each player
        self.players = [] # players

        for _, g in genomes: # (genome id, genome object)
            net = neat.nn.FeedForwardNetwork.create(g, self.config)
            self.nets.append(net)
            player = Player(
                game = self,
                position = pygame.math.Vector2(self.screen.get_width() * 0.3, self.screen.get_height() * 0.5),
                velocity = pygame.math.Vector2(0, 0),
                acceleration = pygame.math.Vector2(0, 0.55),
                size = (50, 50),
                color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            )
            self.players.append(player)
            g.fitness = 0
            self.ge.append(g)

        # setup obstacles
        self.top_obstacles = []
        self.bottom_obstacles = []
        self.spawn_obstacle_pair()
        self.next_obstacle_idx = 0

        # setup timer
        self.spawn_obstacle = pygame.USEREVENT + 1
        pygame.time.set_timer(self.spawn_obstacle, 2000)
        
        # reset game state
        self.is_simulation_started = True

        # reset score
        self.score = 0

        # disable display resizing
        self.screen = pygame.display.set_mode(self.screen_size)

    def handle_game_over(self, player_idx):
        # decrease player fitness
        self.ge[player_idx].fitness -= 1

        # remove player
        self.players.pop(player_idx)
        self.nets.pop(player_idx)
        self.ge.pop(player_idx)

    def create_obstacle(self, position_y, is_bottom):
        return Obstacle(
            game = self,
            position = pygame.math.Vector2(self.screen.get_width(), position_y),
            velocity = pygame.math.Vector2(-3, 0),
            size = (80, 1000),
            color = (0, 0, 0),
            is_bottom = is_bottom
        )

    def remove_obstacles(self):
        old_size = len(self.top_obstacles)
        self.top_obstacles = [top_obstacle for top_obstacle in self.top_obstacles if top_obstacle.rect.right > 0]
        self.bottom_obstacles = [bottom_obstacle for bottom_obstacle in self.bottom_obstacles if bottom_obstacle.rect.right > 0]
        removed_count = old_size - len(self.top_obstacles)
        if removed_count > 0:
            self.next_obstacle_idx -= max(0, self.next_obstacle_idx - removed_count)

    def is_player_colliding(self, player):
        is_out_of_bounds = player.rect.top < 0 or player.rect.bottom > self.screen.get_height()
        is_colliding_with_obstacle = player.rect.collidelist(self.top_obstacles) > -1 or player.rect.collidelist(self.bottom_obstacles) > -1
        return is_out_of_bounds or is_colliding_with_obstacle

    def spawn_obstacle_pair(self):
        gap = 180
        center_y = random.uniform(self.screen.get_height() / 2 - 250, self.screen.get_height() / 2 + 250)
        top_obstacle = self.create_obstacle(center_y - gap / 2, False)
        bottom_obstacle = self.create_obstacle(center_y + gap / 2, True)

        self.top_obstacles.append(top_obstacle)
        self.bottom_obstacles.append(bottom_obstacle)
        
    def display_title_text(self):
        display_text(
            surf = self.screen,
            text = 'FLAPPY BIRD',
            size = 250,
            position = (self.screen.get_width() / 2, self.screen.get_height() / 2),
            color = (0, 0, 0),
        )

    def display_start_simulation_text(self):
        display_text(
            surf = self.screen,
            text = 'Click or press space to start simulation',
            size = 40,
            position = (self.screen.get_width() / 2, self.screen.get_height() / 2 + 200), 
            color = (200, 200, 200),
        )
            
    def display_score(self):
        display_text(
            surf = self.screen,
            text = f'{self.score}',
            size = 600,
            position = (self.screen.get_width() / 2, self.screen.get_height() / 2),
            color = (230, 230, 230)
        )
    
    def display_generation_num(self):
        display_text(
            surf = self.screen,
            text = f'Generation: {self.generation_num}',
            size = 50,
            position = (130, 30),
            color = (0, 0, 0)
        )

    def display_num_alive(self):
        display_text(
            surf = self.screen,
            text = f'Alive: {len(self.players)}',
            size = 50,
            position = (85, 70),
            color = (0, 0, 0)
        )

    # fitness function
    def eval_genomes(self, genomes, config): 
        # initialize genome fitnesses
        for _, genome in genomes:
            genome.fitness = 0
        
        # setup simulation
        self.simulation_setup(genomes)

        # increment generation number
        self.generation_num += 1

        # game loop
        while True:
            # update delta time
            self.delta_time = time.perf_counter() - self.last_time
            self.delta_time *= 60
            self.last_time = time.perf_counter()

            # event loop
            for event in pygame.event.get():
                # handle closing window
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                # handle obstacle spawning
                if event.type == self.spawn_obstacle:
                    self.spawn_obstacle_pair()

            # clear screen
            self.screen.fill(self.screen_bg_color)

            # stop running if all players are dead
            if not self.players:
                break

            # display score
            self.display_score()

            # update players
            for player_idx, player in enumerate(self.players):
                # update player
                player.update(self.screen)
                self.ge[player_idx].fitness += 0.01

                # feed nn inputs (player y pos, dist from top, dist from bottom)
                top_obstacle = self.top_obstacles[self.next_obstacle_idx]
                bottom_obstacle = self.bottom_obstacles[self.next_obstacle_idx]
                top_obstacle_ydist = abs(player.rect.centery - top_obstacle.rect.bottom)
                bottom_obstacle_ydist = abs(player.rect.centery - bottom_obstacle.rect.top)
                horizontal_dist = abs(player.rect.left - top_obstacle.rect.right)
                output = self.nets[player_idx].activate((player.rect.centery, horizontal_dist, top_obstacle_ydist, bottom_obstacle_ydist))
                if output[0] > 0.5:
                    player.jump()

            # update obstacles
            for top_obstacle, bottom_obstacle in zip(self.top_obstacles, self.bottom_obstacles):
                top_obstacle.update(self.screen)
                bottom_obstacle.update(self.screen)
            self.remove_obstacles()

            # display generation number and alive count
            self.display_generation_num()
            self.display_num_alive()

            # check if player is colliding
            for player_idx in range(len(self.players)-1, -1, -1):
                player = self.players[player_idx]
                if self.is_player_colliding(player):
                    self.handle_game_over(player_idx)

            # end simulation if fitness threshold is surpassed
            # fitnesses = [self.ge[i].fitness for i in range(len(self.players))]
            # if fitnesses:
            #     max_fitness = max(fitnesses)
            #     print(f'{max_fitness:.2f}, {self.next_obstacle_idx}')
            #     if max_fitness >= config.fitness_threshold:
            #         break
            # end simulation if score > 15
            if self.score > 15:
                break

            # handle player scoring
            obstacle = self.top_obstacles[self.next_obstacle_idx]
            for player_idx, player in enumerate(self.players):
                if not obstacle.is_passed and player.rect.left >= obstacle.rect.right:
                    # add obstacle to players passed obstacle set
                    obstacle.is_passed = True

                    # increase player fitnesses
                    self.ge[player_idx].fitness += 10

            # increment next obstacle idx and increment score
            if obstacle.is_passed:
                self.next_obstacle_idx += 1
                self.score += 1

            pygame.display.update()
            self.clock.tick(self.max_fps)

    def run_simulation(self):
        winner = self.population.run(self.eval_genomes, 20)

        # show final stats
        print('\nBest genome:\n{!s}'.format(winner))

        # visualize the results
        visualize.plot_stats(self.stats, ylog=False, view=True)
        visualize.plot_species(self.stats, view=True)
        visualize.draw_net(self.config, winner, view=True)


if __name__ == '__main__':
    simulation = Simulation()
    simulation.run_simulation()