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
    
    
    # 4. Create Method dynamically from config
    # Example: config.method_name = "asif" or "csa"
    methods = {}
    for method_name in config.methods:
        MethodClass = get_method_class(method_name)
        methods[method_name] = MethodClass

    # 3. Create Benchmark
    benchmark = MMA_Benchmark(tasks=tasks, methods=methods, config=config)
    # 5. Run Benchmark
    results = benchmark.run(model=None, support_embeddings=support_embeddings)
    
    print(f"Results:", results)

if __name__ == "__main__":
    test_framework()
