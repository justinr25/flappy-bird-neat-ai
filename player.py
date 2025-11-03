import pygame

class Player():
    def __init__(self, game, position, velocity, acceleration, size, color):
        self.game = game
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.size = size
        self.color = color
        self.isJumpButtonReleased = True
        self.passed_obstacles = set()

        self.rect = pygame.Rect((0, 0), size)
        self.rect.center = position

    def jump(self):
        self.velocity.y = -10

    def draw(self, surf):
        pygame.draw.ellipse(surf, self.color, self.rect)

    def update(self, surf):
        # draw player
        self.draw(surf)
        
        # update kinematics values
        self.velocity += self.acceleration * self.game.delta_time
        self.position += self.velocity
        self.rect.center = self.position

