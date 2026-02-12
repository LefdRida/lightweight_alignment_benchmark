from data.dataset_base import EmbeddingDataset, DatasetBase
import numpy as np
from typing import List, Dict, Tuple
from omegaconf import DictConfig
import polars as pl

class Imagenet1k(DatasetBase):
    def __init__(self, root: str, loc_val_solution: str, loc_synset_mapping: str, **kwargs):
        DatasetBase.__init__(self)
        self.load_data(root, loc_val_solution, loc_synset_mapping)

    def load_data(
        self, root: str, loc_val_solution: str,loc_synset_mapping: str
    ) -> None:
        """
        Load the ImageNet dataset from Hugging Face Arrow format and
        corresponding precomputed embeddings.

        Args:
            cfg_dataset: configuration file (expects cfg_dataset.paths.dataset_path)
        
        Returns:
            img_path: dummy list of image identifiers (or placeholder paths)
            mturks_idx: array of image labels (same as orig_idx, kept for compatibility)
            orig_idx: ground truth class indices (int)
            clsidx_to_labels: a dict of class idx to str.
        """
        """
        Args:
            cfg_dataset: configuration file

        Returns:
            img_paths: list of image absolute paths
            text_descriptions: list of text descriptions
        """
        #'/kaggle/input/imagenet-object-localization-challenge/LOC_val_solution.csv'
        #"/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt"
        mapping = {}
        with open(loc_synset_mapping, "r") as f:
            for idx, l in enumerate(f.readlines()):
                class_id = l.split(" ")[0]
                class_name = " ".join(l.split(" ")[1:]).split(",")[0].strip().removesuffix(',')
                mapping[class_id] = (class_name, idx)
                
        table = pl.read_csv(loc_val_solution)
        table = table.with_columns(
            pl.col('PredictionString')
            .map_elements(lambda x: x.split(' ')[0], return_dtype=pl.String)
            .alias('class_string')
            )
        table = table.with_columns(
            pl.col('class_string')
            .map_elements(lambda x: mapping[x][0], return_dtype=pl.String)
            .alias('label')
            )
        table = table.with_columns(
            pl.col('class_string')
            .map_elements(lambda x: mapping[x][1], return_dtype=pl.Int32)
            .alias('label_id')
            )
        table = table.with_columns(
            pl.col('ImageId')
            .map_elements(lambda x: f"{root}/{x}.JPEG", return_dtype=pl.String)
            .alias('image_path')
            )
        
        table = table.with_columns(
            pl.col('label')
            .map_elements(lambda x: f"This is an image of {x}", return_dtype=pl.String)
            .alias('label_description')
            )
        
        self.table = table.sort("label_id")

        self.clsidx_to_labels = {}
        for sample in self.table.select(['label_id', 'label']).unique().to_dicts():
            if sample["label_id"] not in self.clsidx_to_labels:
                self.clsidx_to_labels[sample["label_id"]] = sample["label"]
        

class Imagenet1kZeroshotClassificationDataset(Imagenet1k, EmbeddingDataset):
    def __init__(self, task_config: DictConfig):
        Imagenet1k.__init__(
            self, 
            root=task_config.root, 
            loc_val_solution=task_config.loc_val_solution, 
            loc_synset_mapping=task_config.loc_synset_mapping
            )
        
        self.image_paths = self.table.select("image_path").to_numpy()
        if task_config.generate_embedding:
            self.labels_descriptions = self.table.select("label_description").unique().to_numpy()
            
        else:
            EmbeddingDataset.__init__(
                self,
                img_encoder=task_config.img_encoder,
                text_encoder=task_config.text_encoder,
                hf_img_embedding_name=task_config.hf_img_embedding_name,
                hf_text_embedding_name=task_config.hf_text_embedding_name,
                hf_repo_id=task_config.hf_repo_id,
                train_test_ratio=task_config.train_test_ratio,
                seed=task_config.seed,
                split=task_config.split
            )
            self.labels = self.table.select("label_id").to_numpy()
            self.labels_emb = None
            self.metatask = task_config.metatask
            
    def set_training_paired_embeddings(self) -> None:        

        """Get the paired embeddings for both modalities."""
        if self.text_embeddings.shape[0] != len(self.clsidx_to_labels):
            self.text_embeddings = np.unique(self.text_embeddings, axis=0)
        #, "To pair embeddings, the text embeddings should contain only all possible labels."
        assert self.image_embeddings.shape[0] == len(self.labels), "Each image should have a corresponding list of labels."
        assert self.train_idx is not None or self.split=="train", "Please get the train/test split index first."
        
        text_emb = []

        if self.split == "train":
            for idx, label in enumerate(self.labels):
                label_emb = self.text_embeddings[label].reshape(-1)
                text_emb.append(label_emb)
            train_text_embeddings = np.array(text_emb)

        elif self.split == "large" and self.train_idx is not None:
            train_image_embeddings =  self.image_embeddings[self.train_idx]
            for idx in self.train_idx:
                label = self.labels[idx]
                label_emb = self.text_embeddings[label].reshape(-1)
                text_emb.append(label_emb)
            train_text_embeddings = np.array(text_emb)
            
        else:
            raise ValueError("Please set split to 'train' or get the train/test split index first.")
        assert train_image_embeddings.shape[0] == train_text_embeddings.shape[0]

        self.support_embeddings["train_image"] = train_image_embeddings
        self.support_embeddings["train_text"] = train_text_embeddings
    
    def set_labels_emb(self) -> None:
        """Get the text embeddings for all possible labels."""
        if self.text_embeddings.shape[0] == len(self.clsidx_to_labels):
            labels_emb = self.text_embeddings
        else:
            label_emb = []
            for label_idx in self.clsidx_to_labels:
                # find where the label is in the train_idx
                label_idx_in_ds = np.where(self.labels == label_idx)[0]
                label_emb.append(self.text_embeddings[label_idx_in_ds[0]])
            labels_emb = np.array(label_emb)
            assert labels_emb.shape[0] == len(self.clsidx_to_labels)
        self.support_embeddings["labels_emb"] = labels_emb
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.image_embeddings is not None and self.text_embeddings is not None, "Please load the data first."
        if self.split == "large":
            assert self.train_idx is not None and self.val_idx is not None, "Please get the train/test split index first."
            val_image_embeddings = self.image_embeddings[self.val_idx]
            val_labels = self.labels[self.val_idx] 
        elif self.split == "val":
            val_image_embeddings = self.image_embeddings
            val_labels = self.labels
        else:   
            raise ValueError("Please set split to 'train', 'val' or 'large'.")
        return val_image_embeddings, val_labels