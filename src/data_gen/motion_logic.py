import os
import numpy as np
from dataclasses import dataclass


@dataclass
class ClipRequest:
    room: str = 'warehouse'
    num_clips: int = 1
    num_cams: int = 7
    egomotion: str = 'steer_right'


class ClipsGenerator:

    def __init__(self,):

        self.save_dir = os.path.expanduser('~/data/lora_phys')

        self.rooms = ['warehouse', 'empty']

    def generate(self, request: ClipRequest):
        return

        
