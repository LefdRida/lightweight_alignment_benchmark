import numpy as np
from benchmark import MMA_Benchmark
from metatasks.classification import ClassificationTask
from metatasks.retrieval import RetrievalTask
from methods import get_method_class
from data.loader import load_dataset_metatask
from config import config
from omegaconf import OmegaConf


config = OmegaConf.create(config)

def test_framework():
    
    # 2. Create Tasks
    tasks = []
    for task in config.tasks:
        tasks.append(load_dataset_metatask(task, config))
    support_embeddings = None
    # 3. Create Benchmark
    benchmark = MMA_Benchmark(tasks=tasks)
    
    # 4. Create Method dynamically from config
    # Example: config.method_name = "asif" or "csa"
    for method_name in config.methods:
        MethodClass = get_method_class(method_name)
        method = MethodClass(**config[method_name])
        
        # 5. Run Benchmark
        results = benchmark.run(method=method, model=None, support_embeddings=support_embeddings)
        
        print(f"{method_name.upper()} Results:", results)

if __name__ == "__main__":
    test_framework()
