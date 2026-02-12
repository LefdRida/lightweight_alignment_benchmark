from collections import defaultdict
import json
from typing import Dict, List, Tuple
import numpy as np
from data.dataset_base import DatasetBase, EmbeddingDataset
import polars as pl
from omegaconf import DictConfig


class MScoco(DatasetBase):
    def __init__(self, data_path: str):
        super().__init__()
        self.load_data(data_path)
        
    def load_data(
        self, data_path: str
    ) -> None:
        """
        Load the MSCOCO dataset from local JSON files.

        Returns:
            img_paths: list of image absolute paths
            text_descriptions: list of text descriptions
            categories: list of list of category names per image
        """

        
        with open(f'{data_path}/annotations/captions_train2017.json', 'rb') as f:
            data = json.load(f)
        f.close()

        caption_image_table = (
            pl.DataFrame(data['images'])
            .with_columns(
                pl.col("file_name").map_elements(
                    lambda x: f"{data_path}/train2017/{x}"
                ).alias("image_path")
            )
            .select(['id', 'image_path'])
        )

        caption_table = pl.DataFrame(data['annotations'])
        caption_table = caption_table.join(
            caption_image_table,
            left_on="image_id",
            right_on="id",
            how="inner",
            maintain_order=True
        )

        self.caption_table = (
            caption_table
            .group_by('image_id', 'image_path', maintain_order=True)
            .agg(
                pl.col('caption').alias('captions'),
                pl.col('id')
            )
            .filter(pl.col('captions').arr.lengths() == 5)
        )
        

        with open(f'{data_path}/annotations/instances_train2017.json', 'rb') as f:
            data = json.load(f)
        f.close()
        for cat in data['categories']:
            self.clsidx_to_labels[cat['id']] = cat['name']

        instance_table = (pl.DataFrame(data['annotations'])
                          .filter(pl.col('category_id').is_not_null())
                          .with_columns(pl.col('category_id').alias('label'))
                          .with_columns(
                              pl.col('category_id').map_elements(
                                  lambda x: "This is an image of " + self.clsidx_to_labels[x]
                                  ).alias('label_description')
                          )
        )
        instance_image_table = pl.DataFrame(data['images'])
        instance_table = instance_table.join(
            instance_image_table,
            left_on="image_id",
            right_on="id",
            how="inner",
            maintain_order=True
        )
        self.instance_table = (instance_table
                          .group_by('image_id', 'image_path', maintain_order=True)
                          .agg(pl.col('label').unique(),
                               pl.col('label_description').unique(),
                               pl.col('id')
                            )
        )

               
class MScocoMultiLabelClassificationDataset(MScoco, EmbeddingDataset):
    def __init__(self, task_config: DictConfig):
        MScoco.__init__(
            self, 
            data_path=task_config.data_path
            )
        
        if task_config.generate_embedding:
            self.image_paths = self.instance_table.select("image_path").to_list()
            self.labels_descriptions = (
                self.instance_table
                .select("label_description" , "labels")
                .explode("labels")
                .explode("label_description")
                .sort("labels")
                .select("label_description")
                .unique()
                .to_list()
            )
        else: 
            EmbeddingDataset.__init__(
                self,
                split=task_config.split  
            )
            self.labels_emb = None
            self.labels = self.instance_table.select("label").to_list()
            self.load_two_encoder_data(
                hf_repo_id=task_config.hf_repo_id, 
                hf_img_embedding_name=task_config.hf_img_embedding_name, 
                hf_text_embedding_name=task_config.hf_text_embedding_name
            )
            self.set_labels_emb()
            if task_config.split == "train" or task_config.split == "large":
                self.set_train_test_split_index(
                    train_test_ratio=task_config.train_test_ratio, 
                    seed=task_config.seed
                )
                self.set_training_paired_embeddings()

    def get_labels_emb(self) -> None:
        """Get the text embeddings for all possible labels."""
        if self.text_embeddings.shape[0] == len(self.clsidx_to_labels):
            self.labels_emb = self.text_embeddings
        else:
            labels = [l 
                    for label_list in self.labels
                    for l in label_list
                    ]
            labels = np.array(labels)
            label_emb = []
            for label_idx in self.clsidx_to_labels:
                label_idx_in_ds = np.where(labels == label_idx)[0]
                label_emb.append(self.text_emb[label_idx_in_ds[0]])
            self.labels_emb = np.array(label_emb)
        assert self.labels_emb.shape[0] == len(self.clsidx_to_labels)
        self.support_embeddings["labels_emb"] = self.labels_emb

    def get_training_paired_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the paired embeddings for both modalities."""
        assert self.image_embeddings.shape[0] == len(self.labels), "Each image should have a corresponding list of labels."
        assert self.train_idx is not None or self.split=="train", "Please get the train/test split index first."
        assert self.support_embeddings.get('labels_emb', None) is not None, "Please get the labels embeddings first."
        
        text_emb = []
        image_emb = []
        self.labels_emb = self.support_embeddings['labels_emb']

        if self.split == "train":
            for idx, label_list in enumerate(self.labels):
                for label_idx in label_list:
                    label_emb = self.text_embeddings[label_idx].reshape(1, -1)
                    text_emb.append(label_emb)
                    image_emb.append(self.image_embeddings[idx].reshape(1, -1))
            train_image_embeddings = np.array(image_emb)
            train_text_embeddings = np.array(text_emb)

        elif self.split == "large" and self.train_idx is not None:
            for idx in self.train_idx:
                label_list = self.labels[idx]
                for label_idx in label_list:
                    label_emb = self.text_embeddings[label_idx].reshape(1, -1)
                    text_emb.append(label_emb)
                    image_emb.append(self.image_embeddings[idx].reshape(1, -1))
            train_image_embeddings = np.array(image_emb)
            train_text_embeddings = np.array(text_emb)
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
            val_labels = self.labels[self.val_idx] 
        elif self.split == "val":
            val_image_embeddings = self.image_embeddings
            val_labels = self.labels
        else:   
            raise ValueError("Please set split to 'train', 'val' or 'large'.")

        return val_image_embeddings, val_labels


class MScocoRetrievalDataset(MScoco, EmbeddingDataset):
    def __init__(self, task_config: DictConfig):
        MScoco.__init__(
            self, 
            data_path=task_config.data_path
            )
        if task_config.generate_embedding:
            self.captions = self.caption_table.select("captions").explode("captions").to_list()
            self.image_paths = self.caption_table.select("image_path").to_list()
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
            self.image_paths = self.caption_table.select("image_path").to_list()
            self.captions = self.caption_table.select("captions").to_list()
            self.num_caption_per_image = 5
            self.num_image_per_caption = 1
            self.gt_img_doc_ids = [idx for idx in range(len(self.image_paths)) for _ in range(self.num_caption_per_image)]      
            self.gt_caption_doc_ids = list(range(len(self.captions)))
            self.load_two_encoder_data(
                hf_repo_id=task_config.hf_repo_id, 
                hf_img_embedding_name=task_config.hf_img_embedding_name, 
                hf_text_embedding_name=task_config.hf_text_embedding_name
            )
            if task_config.split == "train" or task_config.split == "large":
                self.set_train_test_split_index(
                    train_test_ratio=task_config.train_test_ratio, 
                    seed=task_config.seed
                )
                self.set_training_paired_embeddings()
        
    def get_training_paired_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the paired embeddings for both modalities."""
        assert self.text_embeddings.shape[0] == len(self.captions)*5, "To pair embeddings, the text embeddings should contain only all possible labels."
        assert self.image_embeddings.shape[0] == len(self.captions), "Each image should have a corresponding list of labels."
        assert self.train_idx is not None or self.split=="train", "Please get the train/test split index first."
        
        text_emb = []
        image_emb = []

        if self.split == "train":
            for idx, caption_list in enumerate(self.captions):
                caption_emb = self.text_embeddings[idx*5:(idx+1)*5].reshape(5, -1)
                text_emb.append(caption_emb)
                image_emb.append(
                    np.repeat(self.image_embeddings[idx].reshape(1, -1), 5, axis=0 )
                    )
            train_image_embeddings = np.array(image_emb)
            train_text_embeddings = np.array(text_emb)

        elif self.split == "large" and self.train_idx is not None:
            for idx in self.train_idx:
                caption_emb = self.text_embeddings[idx*5:(idx+1)*5].reshape(5, -1)
                text_emb.append(caption_emb)
                image_emb.append(
                    np.repeat(self.image_embeddings[idx].reshape(1, -1), 5, axis=0 )
                    )
            train_image_embeddings = np.array(image_emb)
            train_text_embeddings = np.array(text_emb)
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