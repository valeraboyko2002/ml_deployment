import pickle
import numpy as np
from typing import Dict, Any
import time 
from functools import lru_cache

class MLModelWrapper:
    """Обертка для ml модели с кэш. и мониторингом"""

    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.predictions_made = 0
        self.total_inference_time = 0

    def _load_model(self,model_path:str):
        """загрузка модели из файла"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    @lru_cache(maxsize=1001)
    def predict_cached(self,input_hash: str, features: tuple):
        """кэширование предсказаеие лоя одинаковых запросов"""
        return self.model.predict([features])[0]
    
    def predict(self,data: Dict[str,Any]):
        """Основной метод предсказания с метриками"""
        start_time = time.time()

        try: