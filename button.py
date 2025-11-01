import pygame

class Button:
    def __init__(self, game, text, size, position, text_color, bg_color=None, font=None):
        self.game = game
        self.text = text
        self.size = size
        self.position = position
        self.text_color = text_color
        self.bg_color = bg_color
        self.font = pygame.font.Font(font, size)

        self.surf = self.font.render(text, True, text_color, bg_color)
        self.rect = self.surf.get_rect(center = position)
    
    def is_hovered(self):
        return self.rect.collidepoint(pygame.mouse.get_pos())
    
    def is_clicked(self):
        return pygame.mouse.get_pressed()[0] and self.is_hovered()

    def hover_event(self):
        self.surf = self.font.render(self.text, True, self.text_color, (240, 240, 240))

    def click_event(self):
        print('click event')
        
    def draw(self):
        self.game.screen.blit(self.surf, self.rect)

    def update(self):
        self.draw()

        # handle click event
        if self.is_clicked():
            self.click_event()
        
        # handle hover event
        if self.is_hovered():
            self.hover_event()
        else:
            self.surf = self.font.render(self.text, True, self.text_color, self.bg_color)
            
        