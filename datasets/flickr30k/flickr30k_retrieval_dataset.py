from data.dataset_base import EmbeddingDataset, DatasetBase
import numpy as np
from typing import List, Dict, Tuple
from omegaconf import DictConfig
from pathlib import Path
import polars as pl


class Flickr30k(DatasetBase):
    def __init__(self, dataset_path: str):
        super().__init__()
        self.captions = None
        self.num_caption_per_image = None
        self.num_image_per_caption = None
        self.gt_caption_doc_ids = None
        self.gt_img_doc_ids = None
        self.load_data(dataset_path)
        
            
    def load_data(
    self, dataset_path:str
    ) -> tuple[list[str], list[str], np.ndarray, list[str]]:
        """Load the Flickr dataset (https://huggingface.co/datasets/nlphuji/flickr30k).

        Args:
            cfg_dataset: configuration file

        Returns:
            img_paths: list of image absolute paths
            text_descriptions: list of text descriptions
            splits: list of splits [train, test, val] (str)
            obj_ids: list of object ids (str)
        """
        # load Flickr train json filee. columns: [raw, sentids, split, filename, img_id]
        self.flickr = (
            pl.read_csv(Path(dataset_path) / 'captions.txt', separator=',')
            .with_columns(pl.col('caption').str.len_chars().alias('len'))
            .sort('len', descending=True)
            .group_by('image', maintain_order=True)
            .agg(pl.col('caption'))
            .with_columns(
                pl.col("image").map_elements(
                    lambda x: str(Path(dataset_path) / "Images" / x), return_dtype=pl.String
                ).alias("image_path")
            )
        )


class Flickr30kRetrievalDataset(Flickr30k, EmbeddingDataset):
    def __init__(self, task_config: DictConfig):
        Flickr30k.__init__(self, dataset_path=task_config.dataset_path)
        
        self.image_paths = self.flickr.select("image_path").to_series().to_list()
        if task_config.generate_embedding:
            self.captions = self.flickr.select("caption").explode("caption").to_series().to_list()
        else:
            EmbeddingDataset.__init__(
                self,
                split=task_config.split
            )
            self.captions = self.flickr.select("caption").to_series().to_list()
            self.num_caption_per_image = 5
            self.num_image_per_caption = 1
            self.load_two_encoder_data(
                hf_repo_id=task_config.hf_repo_id, 
                hf_img_embedding_name=task_config.hf_img_embedding_name, 
                hf_text_embedding_name=task_config.hf_text_embedding_name
            )
            if self.split == "train" or self.split == "large":
                self.set_train_test_split_index(
                    train_test_ratio=task_config.train_test_ratio, 
                    seed=task_config.seed
                )
                self.get_training_paired_embeddings()
    

    def get_training_paired_embeddings(self) -> None:
        """Get the paired embeddings for both modalities."""
        assert self.text_embeddings.shape[0] == len(self.captions)*5, "To pair embeddings, the text embeddings should contain only all possible labels."
        assert self.image_embeddings.shape[0] == len(self.captions), "Each image should have a corresponding list of labels."
        assert self.train_idx is not None or self.split=="train", "Please get the train/test split index first."
        
        text_emb = []
        image_emb = []
        print(self.text_embeddings.shape)
        print(self.image_embeddings.shape)
        if self.split == "train":
            for idx, caption_list in enumerate(self.captions):
                caption_emb = self.text_embeddings[idx*5:(idx+1)*5].reshape(5, -1)
                text_emb.append(caption_emb)
                image_emb.append(
                    np.repeat(self.image_embeddings[idx].reshape(1, -1), 5, axis=0 )
                    )
            train_image_embeddings = np.concatenate(image_emb, axis=0)
            train_text_embeddings = np.concatenate(text_emb, axis=0)

        elif self.split == "large" and self.train_idx is not None:
            for idx in self.train_idx:
                caption_emb = self.text_embeddings[idx*5:(idx+1)*5].reshape(5, -1)
                text_emb.append(caption_emb)
                image_emb.append(
                    np.repeat(self.image_embeddings[idx].reshape(1, -1), 5, axis=0 )
                    )
            train_image_embeddings = np.concatenate(image_emb, axis=0)
            train_text_embeddings = np.concatenate(text_emb, axis=0)
        else:
            raise ValueError("Please set split to 'train' or get the train/test split index first.")
        assert train_image_embeddings.shape[0] == train_text_embeddings.shape[0]
        self.support_embeddings["train_image"] = train_image_embeddings
        self.support_embeddings["train_text"] = train_text_embeddings

    def get_test_data(self):
        assert self.image_embeddings is not None and self.text_embeddings is not None, "Please load the data first."
        if self.split == "large":
            assert self.train_idx is not None and self.val_idx is not None, "Please get the train/test split index first."
            val_image_embeddings = self.image_embeddings[self.val_idx]
            self.gt_img_doc_ids = [idx for idx in range(len(val_image_embeddings)) for _ in range(self.num_caption_per_image)]
            val_text_idx = np.array([
                idx for i in self.val_idx 
                for idx in range(i*self.num_caption_per_image, (i+1)*self.num_caption_per_image)
            ])
            val_text_embeddings = self.text_embeddings[val_text_idx]
            self.gt_caption_doc_ids = list(range(len(val_text_embeddings)))

        elif self.split == "val":
            val_image_embeddings = self.image_embeddings
            self.gt_img_doc_ids = [idx for idx in range(len(val_image_embeddings)) for _ in range(self.num_caption_per_image)]
            val_text_embeddings = self.text_embeddings
            self.gt_caption_doc_ids = list(range(len(val_text_embeddings)))
        else:   
            raise ValueError("Please set split to 'train', 'val' or 'large'.")

        return val_image_embeddings, val_text_embeddings, self.gt_caption_doc_ids, self.gt_img_doc_ids

    
