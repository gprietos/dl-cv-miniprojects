from abc import ABC, abstractmethod
import numpy as np


class BaseLogger(ABC):
    @abstractmethod
    def log(self,
            metrics_dict : dict,
            step : int
            ): 
        raise NotImplementedError
    
    @abstractmethod
    def log_image(self,
                  image : np.array,
                  image_name : str
                  ):        
        raise NotImplementedError
    
    @abstractmethod
    def close(self): 
        raise NotImplementedError