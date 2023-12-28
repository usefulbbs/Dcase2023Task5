## tensorboard 可视化包

import torch
from torch.utils.tensorboard import SummaryWriter

__all__=["Summary"]

class Summary():
    def __init__(self,path) -> None:
        super().__init__() 
        self.writer = SummaryWriter(log_dir=path)
    
    def add_scalar(self,name,x,tag):
        self.writer.add_scalars(name,x,tag)
        
    def add_img(self,name,x,tag):
        self.writer.add_images(name,x,global_step=tag)
    
    def add_text(self,name,x,tag):
        self.writer.add_text(name,x,tag)
    
    def close(self):
        self.writer.close()
        
   