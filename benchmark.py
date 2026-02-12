from typing import List, Union, Dict, Any
from base.base import AbsTask, AbsModel, AbsMethod

class MMA_Benchmark:
    """Main Orchestrator for the MMA Benchmark"""
    
    def __init__(self, tasks: List[AbsTask]):
        self.tasks = tasks

    def run(self, method: AbsMethod, model: AbsModel, support_embeddings: Dict[str, Any] = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Run all tasks in the benchmark for a given method and model."""
        results = {}
        for task in self.tasks:
            print(f"Running task: {task.name} ({task.task_type})")
            results[task.name] = task.run(method, support_embeddings=support_embeddings, **kwargs)
        return results

    def add_task(self, task: AbsTask):
        """Add a new task to the benchmark."""
        self.tasks.append(task)
