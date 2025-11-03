import pygame

class Obstacle():
    def __init__(self, game, position, velocity, size, color, is_bottom):
        self.game = game
        self.position = position
        self.velocity = velocity
        self.size = size
        self.color = color
        self.is_bottom = is_bottom
        self.is_passed = False

        self.rect = pygame.Rect(position, size)

    def draw(self, surf):
        pygame.draw.rect(surf, self.color, self.rect)

    def update(self, surf):
        # draw obstacle
        self.draw(surf)

        # update kinematics values
        # self.rect.move_ip(self.velocity * self.game.delta_time)
        self.position += self.velocity
        if self.is_bottom:
            self.rect.topleft = self.position
        else:
            self.rect.bottomleft = self.position