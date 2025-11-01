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

        self.rect = pygame.Rect((0, 0), size)
        self.rect.center = position

    def jump(self):
        self.velocity.y = -10

    def draw(self, surf):
        pygame.draw.ellipse(surf, self.color, self.rect)

    def update(self, surf):
        # draw player
        self.draw(surf)

        # handle jumping
        keys = pygame.key.get_pressed()
        if self.isJumpButtonReleased and (keys[pygame.K_SPACE] or pygame.mouse.get_pressed()[0]):
            self.jump()
            self.isJumpButtonReleased = False
        if not keys[pygame.K_SPACE] and not pygame.mouse.get_pressed()[0]:
            self.isJumpButtonReleased = True

        # update kinematics values
        self.velocity += self.acceleration * self.game.delta_time
        self.rect.move_ip(self.velocity * self.game.delta_time)
        # self.position += self.velocity
        # self.rect.center = self.position
        