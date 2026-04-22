
import dearpygui.dearpygui as dpg


import numpy as np
import random
import os, sys
import torch

import sys
from scene import Scene
from argparse import ArgumentParser
from utils.timer import Timer


import json


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
import matplotlib.pyplot as plt

from gui_utils.base import GUIBase
class GUI(GUIBase):
    
    def save_config(self):
        with open(os.path.join(self.logs_path, 'config.json'), "w") as fp:
            json.dump(self.config, fp)
                
    def __init__(self,args,data,name):
        self.name = name # project name
        self.logs_path = './logs/'+name # path to log files
        self.args = args 
        self.data_path = data # path to splat

        # Create/Upload config config file
        print("Loading project...")
        if os.path.exists(self.logs_path) == False:
            print('  New project')
            os.makedirs(self.logs_path)
            self.config = {
                "name":name,
                "data":data,
            }
            
            self.save_config()
        elif os.path.exists(os.path.join(self.logs_path, 'config.json')) == False:
            print('  New config')
            self.config = {
                "name":name,
                "data":data,
            }
            self.save_config()

        else:
            print(f'  Loading {self.logs_path}...')
            with open(os.path.join(self.logs_path, 'config.json')) as fp:
                self.config = json.load(fp)
        
        
        # Set the background color
        bg_color = [1, 1, 1] 
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Load the GS scene
        scene = Scene(
            data
        )
        
        # Initialize DPG      
        super().__init__(scene, name)

        # Initialize training
        self.timer = Timer()
        self.timer.start()



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()

    # print('Runing from ... ',os.environ["SLURM_PROCID"])
    # exit()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    parser.add_argument('--data', type=str, default="")
    parser.add_argument('--expname', type=str, default="")
    
    args = parser.parse_args(sys.argv[1:])

    torch.autograd.set_detect_anomaly(True)

    initial_name = args.expname     
    name = f'{initial_name}'
    
    initial_data = args.data     
    data = f'{initial_data}'
    
    gui = GUI(
        args=args,
        data=data,
        name=name,
    )
    
    gui.render()
    del gui
    torch.cuda.empty_cache()
