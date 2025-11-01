import sys
import time
import random
import json

import pygame

from utils import display_text
from player import Player
from obstacle import Obstacle
from reset_button import ResetButton

class Game:
    def __init__(self):
        # setup pygame
        pygame.init()
        pygame.display.set_caption('Flappy Bird')
        self.monitor_size = [pygame.display.Info().current_w, pygame.display.Info().current_h]
        self.is_fullscreen = False
        self.screen_size = 1280, 720
        self.max_fps = 60
        self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.last_time = time.perf_counter()
        self.screen_bg_color = (255, 255, 255)
        self.is_game_played = False
        self.is_game_active = False
        self.high_score = 0
        with open('high_score.json', 'r') as high_score_file:
            high_score_obj = json.load(high_score_file)
            self.high_score = high_score_obj['high_score']
        self.reset_high_score_btn = ResetButton(
            game = self,
            text = 'Reset High Score',
            size = 30,
            position = (100, self.screen.get_height() - 30),
            text_color = (0, 0, 0)
        )

    def game_setup(self):
        # setup player
        self.player = Player(
                game = self,
                position = pygame.math.Vector2(self.screen.get_width() * 0.3, self.screen.get_height() * 0.5),
                velocity = pygame.math.Vector2(0, 0),
                acceleration = pygame.math.Vector2(0, 0.55),
                size = (50, 50),
                color = (0, 0, 0)
            )

        # setup obstacles
        self.top_obstacles = []
        self.bottom_obstacles = []

        # setup timer
        self.spawn_obstacle = pygame.USEREVENT + 1
        pygame.time.set_timer(self.spawn_obstacle, 2000)
        
        # reset game state
        self.is_game_active = True
        self.is_game_played = True

        # reset score
        self.score = 0

        # disable display resizing
        self.screen = pygame.display.set_mode(self.screen_size)

    def save_high_score(self):
        # update high score
        if self.score > self.high_score:
            self.high_score = self.score

        # save high score to json file
        high_score_obj = {'high_score': self.high_score}
        with open('high_score.json', 'w') as high_score_file:
            json.dump(high_score_obj, high_score_file)

    def handle_game_over(self):
        # enable display resizing
        self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)

        # save high score to json file
        self.save_high_score()

    def create_obstacle(self, position_y):
        return Obstacle(
                game = self,
                position = pygame.math.Vector2(self.screen.get_width(), position_y),
                velocity = pygame.math.Vector2(-3, 0),
                size = (80, 1000),
                color = (0, 0, 0)
            )

    def remove_obstacles(self):
        self.top_obstacles = [top_obstacle for top_obstacle in self.top_obstacles if top_obstacle.rect.right > 0]
        self.bottom_obstacles = [bottom_obstacle for bottom_obstacle in self.bottom_obstacles if bottom_obstacle.rect.right > 0]

    def is_player_colliding(self):
        is_out_of_bounds = self.player.rect.top < 0 or self.player.rect.bottom > self.screen.get_height()
        is_colliding_with_obstacle = self.player.rect.collidelist(self.top_obstacles) > -1 or self.player.rect.collidelist(self.bottom_obstacles) > -1
        return is_out_of_bounds or is_colliding_with_obstacle

    def update_score(self):
        for obstacle in self.top_obstacles:
            if self.player.rect.centerx >= obstacle.rect.centerx and not obstacle.passed:
                self.score += 1
                obstacle.passed = True

    def display_score(self):
        display_text(
            surf = self.screen,
            text = f'{self.score}',
            size = 600,
            position = (self.screen.get_width() / 2, self.screen.get_height() / 2),
            color = (230, 230, 230)
        )

    def display_title_text(self):
        display_text(
            surf = self.screen,
            text = 'FLAPPY BIRD',
            size = 250,
            position = (self.screen.get_width() / 2, self.screen.get_height() / 2),
            color = (0, 0, 0),
        )

    def display_play_game_text(self):
        display_text(
            surf = self.screen,
            text = 'Click or press space to play',
            size = 40,
            position = (self.screen.get_width() / 2, self.screen.get_height() / 2 + 200), 
            color = (200, 200, 200),
        )

    def display_game_over_text(self):
        display_text(
            surf = self.screen,
            text = 'GAME OVER',
            size = 200,
            position = (self.screen.get_width() / 2, self.screen.get_height() / 2),
            color = (0, 0, 0)
        )

    def display_game_over_score_text(self):
        display_text(
            surf = self.screen,
            text = f'Score: {self.score}',
            size = 60,
            position = (self.screen.get_width() / 2, self.screen.get_height() / 2 + 120),
            color = (0, 0, 0)
        )

    def display_game_over_high_score_text(self):
        display_text(
            surf = self.screen,
            text = f'High Score: {self.high_score}',
            size = 60,
            position = (self.screen.get_width() / 2, self.screen.get_height() / 2 + 180),
            color = (0, 0, 0)
        )

    def display_play_again_text(self):
        display_text(
            surf = self.screen,
            text = 'Click or press space to play again',
            size = 40,
            position = (self.screen.get_width() / 2, self.screen.get_height() - 70),
            color = (200, 200, 200)
        )

    def display_reset_high_score_btn(self):
       self.reset_high_score_btn.update()

    def run(self):
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
                
                if self.is_game_active:
                    # handle obstacle spawning
                    if event.type == self.spawn_obstacle:
                        gap = 160
                        center_y = random.uniform(self.screen.get_height() / 2 - 250, self.screen.get_height() / 2 + 250)
                        top_obstacle = self.create_obstacle(center_y - gap / 2 - 1000)
                        bottom_obstacle = self.create_obstacle(center_y + gap / 2)

                        self.top_obstacles.append(top_obstacle)
                        self.bottom_obstacles.append(bottom_obstacle)
                else:
                    # handle resizing
                    if event.type == pygame.VIDEORESIZE:
                        if not self.is_fullscreen:
                            self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    
                    # handle toggling fullscreen with f
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                        self.is_fullscreen = not self.is_fullscreen
                        if self.is_fullscreen:
                            self.screen = pygame.display.set_mode(self.monitor_size, pygame.FULLSCREEN)
                        else:
                            self.screen = pygame.display.set_mode((self.screen.get_width(), self.screen.get_height()), pygame.RESIZABLE)

                    # handle resetting display size with r
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)

                    # * DEBUG: toggle framerate with e
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                        if self.max_fps == 60:
                            self.max_fps = 20
                        else:
                            self.max_fps = 60

                    # handle starting game
                    is_space_pressed = event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE
                    is_mouse_clicked = event.type == pygame.MOUSEBUTTONDOWN
                    if not self.reset_high_score_btn.is_clicked() and (is_space_pressed or is_mouse_clicked):
                        self.game_setup()

            # clear screen
            self.screen.fill(self.screen_bg_color)

            if self.is_game_active:
                # update and display score
                self.update_score()
                self.display_score()

                # update player
                self.player.update(self.screen)

                # update obstacles
                for top_obstacle, bottom_obstacle in zip(self.top_obstacles, self.bottom_obstacles):
                    top_obstacle.update(self.screen)
                    bottom_obstacle.update(self.screen)
                self.remove_obstacles()
                
                # check if player is colliding
                self.is_game_active = not self.is_player_colliding()
                if not self.is_game_active:
                    self.handle_game_over()
            else:
                if not self.is_game_played:
                    self.display_title_text()
                    self.display_play_game_text()
                else:
                    self.screen.fill(self.screen_bg_color)

                    self.display_game_over_text()
                    self.display_game_over_score_text()
                    self.display_game_over_high_score_text()
                    self.display_play_again_text()
                    self.display_reset_high_score_btn()

            pygame.display.update()
            self.clock.tick(self.max_fps)

if __name__ == '__main__':
    game = Game()
    game.run()