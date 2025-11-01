import pygame

def display_text(surf, text, size, position, color, font=None):
    text_font = pygame.font.Font(font, size)
    text_surf = text_font.render(text, True, color)
    text_rect = text_surf.get_rect(center = position)
    surf.blit(text_surf, text_rect)