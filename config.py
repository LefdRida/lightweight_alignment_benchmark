config = {
    "tasks": ["flickr30k"], # dataset_name
    "methods": ["asif", "csa"],  # Method to use: "asif", "csa", or "cka"
    "csa":{
        "sim_dim": 250,
    },
    "asif":{
        "non_zeros": 800,
    },
    "retrieval":{
        "topk": 5,
        "num_gt": 5,
    },
    "classification":{
    },
    "support_embeddings": None,

    "imagenet1k": {
        "root": "/home/rida.lefdali/work/ImageNet/val",
        "loc_val_solution": "/home/rida.lefdali/work/ImageNet/LOC_val_solution.csv",
        "loc_synset_mapping": "/home/rida.lefdali/work/ImageNet/LOC_synset_mapping.txt",
        "hf_img_embedding_name": "ImageNet_img_embed_dinov2-giant.pkl", 
        "hf_text_embedding_name": "ImageNet_text_embed_gtr-t5-large.pkl", 
        "hf_repo_id": "ridalefdali/ImageNet_embeddings", 
        "train_test_ratio": 0.7,
        "seed": 42,
        "split": "large",
        "generate_embedding": False,
        "metatask": "classification", # only "classification"
    },
    "flickr30k": {
        "dataset_path": "/home/rida.lefdali/work/dataset/flickr30k",
        "hf_img_embedding_name": "flickr30k_dinov2_dinov2-giant_image_embeddings.pkl", 
        "hf_text_embedding_name": "flickr30k_gtr_t5_gtr-t5-large_text_embeddings.pkl", 
        "hf_repo_id": "ridalefdali/flickr30k_embeddings", 
        "train_test_ratio": 0.7,
        "seed": 42,
        "split": "large",
        "num_caption_per_image": 5,
        "num_image_per_caption": 1,
        "generate_embedding": False,
        "metatask": "retrieval", # only "retrieval"
    },
    "mscoco": {
        "data_path": "",
        "hf_img_embedding_name": "", 
        "hf_text_embedding_name": "", 
        "hf_repo_id": "", 
        "train_test_ratio": 0.8,
        "seed": 42,
        "split": "large",
        "num_caption_per_image": 5,
        "num_image_per_caption": 1,
        "generate_embedding": True,
        "metatask": "retrieval", # "classification" or  "retrieval"
    },
    "embedding_model": {
        "img_encoder": "dinov2", 
        "text_encoder": "gtr_t5", 
        "image_model_variant": "dinov2-giant",
        "text_model_variant": "gtr-t5-large",
        "batch_size": 50,
    }
}