from collections import defaultdict
import json
from typing import Dict, List, Tuple
import numpy as np
from data.dataset_base import DatasetBase, EmbeddingDataset
import polars as pl
from omegaconf import DictConfig


class MScoco(DatasetBase):
    # TODO: Implement The raw class of MSCOCO dataset
    pass
        

               
class MScocoMultiLabelClassificationDataset(MScoco, EmbeddingDataset):
    # TODO: Implement this Class
    pass 

    
class MScocoRetrievalDataset(MScoco, EmbeddingDataset):
    # TODO: Implement this Class
    pass 