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
            .agg(pl.col('caption').alias('captions'))
            .with_columns(
                pl.col("image").map_elements(
                    lambda x: str(Path(dataset_path) / "Images" / x), return_dtype=pl.String
                ).alias("image_path")
            )
            .with_row_index(name="image_id")
            .explode("captions")
            .with_row_index(name="caption_id")
            .group_by("image_id", maintain_order=True)
            .agg(pl.col("image_path").first(), pl.col("captions"), pl.col("caption_id"))
        )


class Flickr30kRetrievalDataset(Flickr30k, EmbeddingDataset):
    def __init__(self, task_config: DictConfig):
        Flickr30k.__init__(self, dataset_path=task_config.dataset_path)
        
        self.image_paths = self.flickr.select("image_path").to_series().to_list()
        if task_config.generate_embedding:
            self.captions = self.flickr.select("captions").explode("captions").to_series().to_list()
        else:
            EmbeddingDataset.__init__(
                self,
                split=task_config.split
            )
            self.captions = self.flickr.select("captions").to_series().to_list()
            self.num_caption_per_image = 5
            self.num_image_per_caption = 1
            self.img_caption_mapping = self.flickr.select("image_id", "caption_id").to_dicts()
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
        assert self.text_embeddings.shape[0] == len(self.captions)*5, \
            "To pair embeddings, the text embeddings should contain only all possible labels."
        assert self.image_embeddings.shape[0] == len(self.captions), \
            "Each image should have a corresponding list of labels."
        assert self.train_idx is not None or self.split=="train", \
            "Please get the train/test split index first."
        
        text_emb = []
        image_emb = []
        if self.split == "train":
            for item in self.img_caption_mapping:
                image_ids = item["image_id"]
                caption_ids = item["caption_id"]
                caption_emb = self.text_embeddings[caption_ids].reshape(len(caption_ids), -1)
                text_emb.append(caption_emb)
                image_emb.append(
                    np.repeat(self.image_embeddings[image_ids].reshape(1, -1), len(caption_ids), axis=0 )
                    )
            train_image_embeddings = np.concatenate(image_emb, axis=0)
            train_text_embeddings = np.concatenate(text_emb, axis=0)
            
        elif self.split == "large" and self.train_idx is not None:
            for idx in self.train_idx:
                image_ids = self.img_caption_mapping[idx]["image_id"]
                caption_ids = self.img_caption_mapping[idx]["caption_id"]
                caption_emb = self.text_embeddings[caption_ids].reshape(len(caption_ids), -1)
                text_emb.append(caption_emb)
                image_emb.append(
                    np.repeat(self.image_embeddings[image_ids].reshape(1, -1), len(caption_ids), axis=0)
                    )
            train_image_embeddings = np.concatenate(image_emb, axis=0)
            train_text_embeddings = np.concatenate(text_emb, axis=0)
        else:
            raise ValueError("Please set split to 'train' or get the train/test split index first.")
        
        assert train_image_embeddings.shape[0] == train_text_embeddings.shape[0]
        self.support_embeddings["train_image"] = train_image_embeddings
        self.support_embeddings["train_text"] = train_text_embeddings

    def get_test_data(self):
        assert self.image_embeddings is not None and self.text_embeddings is not None, \
            "Please load the data first."
        if self.split == "large":
            assert self.train_idx is not None and self.val_idx is not None, \
                "Please get the train/test split index first."
            self.text_to_image_gt_ids = {}
            self.image_to_text_gt_ids = {}
            val_text_idx = []
            for idx, image_id in enumerate(self.val_idx):
                caption_ids = self.img_caption_mapping[image_id]["caption_id"]
                if len(caption_ids) != self.num_caption_per_image:
                    continue
                val_text_idx.extend(caption_ids)
                val_caption_ids = list(
                    range(idx*self.num_caption_per_image, (idx+1)*self.num_caption_per_image)
                    )
                self.image_to_text_gt_ids[idx] = val_caption_ids
                for val_caption_id in val_caption_ids:
                    self.text_to_image_gt_ids[val_caption_id] = [idx]
            val_image_embeddings = self.image_embeddings[self.val_idx]
            val_text_embeddings = self.text_embeddings[val_text_idx]

        elif self.split == "val":
            val_image_embeddings = self.image_embeddings
            val_text_embeddings = self.text_embeddings
            for item in enumerate(self.img_caption_mapping):
                image_ids = item["image_id"]
                caption_ids = item["caption_id"]
                self.image_to_text_gt_ids[image_ids] = caption_ids
                for caption_id in caption_ids:
                    self.text_to_image_gt_ids[caption_id] = [image_ids] 
        else:   
            raise ValueError("Please set split to 'train', 'val' or 'large'.")

        return val_image_embeddings, val_text_embeddings, self.image_to_text_gt_ids, self.text_to_image_gt_ids

    def get_support_embeddings(self):
        return self.support_embeddings
