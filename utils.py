import pygame
import math

def display_text(surf, text, size, position, color, font=None):
    text_font = pygame.font.Font(font, size)
    text_surf = text_font.render(text, True, color)
    text_rect = text_surf.get_rect(center = position)
    surf.blit(text_surf, text_rect)

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dx = x2-x1
    dy = y2-y1
    return math.hypot(dx, dy)