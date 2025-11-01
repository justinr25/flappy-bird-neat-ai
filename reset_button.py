import json

import pygame

from button import Button

class ResetButton(Button):
    def __init__(self, game, text, size, position, text_color, bg_color=None, font=None):
        super().__init__(game, text, size, position, text_color, bg_color, font)
    
    def click_event(self):
        # save high score of 0 to json file
        with open('high_score.json', 'w') as high_score_file:
            json.dump({'high_score': 0}, high_score_file)

        # reset high score for player
        self.game.high_score = 0