from models import get_image_embedding_model, get_text_embedding_model
from huggingface_hub import HfApi, create_repo, upload_file
from .dataset_utils import upload_embeddings_to_hf
from .embed_data import embed_images, embed_text
from data import get_dataset_class, list_datasets
from omegaconf import DictConfig

HF_TOKEN = None

def get_data_embedding(ds_name:str, task_config: DictConfig, embedding_model_config: DictConfig):
    
    repo_id = task_config.hf_repo_id
    create_repo(repo_id=repo_id, exist_ok=True, token=HF_TOKEN)

    embeddings_saving_path = "temp_embeddings.pkl"

    api = HfApi(token=HF_TOKEN)
    batch_size: int = embedding_model_config.batch_size
    image_encoder_name: str = embedding_model_config.img_encoder
    text_encoder_name: str = embedding_model_config.text_encoder
    image_model_variant: str = embedding_model_config.image_model_variant
    text_model_variant: str = embedding_model_config.text_model_variant
    metatask: str = task_config.metatask 

    try:
        DatasetClass = get_dataset_class(f"{ds_name}-{metatask}-embedding")
    except ValueError:
        print(f"Dataset {ds_name}-{metatask} not found. Available: {list_datasets()}")
        exit(1)

    # Instantiate dataset
    try:
        dataset = DatasetClass(task_config)
    except TypeError as e:
        print(f"Error initializing {ds_name}: {e}")
        print("Please check the script and provide necessary paths in 'dataset_args'.")
        exit(1)
    
    # Embed Images
    if hasattr(dataset, 'image_paths'):
        print(f"Embedding images for {ds_name}...")
        image_embedding, total_params = embed_images(
            image_paths=dataset.get_image_paths(), 
            image_encoder=get_image_embedding_model(image_encoder_name), 
            batch_size=batch_size, 
            model_variant=image_model_variant
            )
        
        upload_embeddings_to_hf(
            embeddings=image_embedding, 
            embeddings_saving_path=embeddings_saving_path, 
            hf_api=api, 
            repo_id=repo_id, 
            path_in_repo=f"{ds_name}_{image_encoder_name}_{image_model_variant}_image_embeddings.pkl"
        )
    
    # Embed Text (Labels or Captions)
    if hasattr(dataset, 'labels_descriptions'):
        print(f"Embedding labels/descriptions for {ds_name}...")
        text_embedding, total_params = embed_text(
            text=dataset.get_labels_descriptions(), 
            text_encoder=get_text_embedding_model(text_encoder_name), 
            batch_size=batch_size, 
            model_variant=text_model_variant
            )
        upload_embeddings_to_hf(
            embeddings=text_embedding, 
            embeddings_saving_path=embeddings_saving_path, 
            hf_api=api,     
            repo_id=repo_id, 
            path_in_repo=f"{ds_name}_{text_encoder_name}_{text_model_variant}_text_embeddings.pkl"
        )
    
    elif hasattr(dataset, 'captions'):
        print(f"Embedding captions for {ds_name}...")
        text_embedding, total_params = embed_text(
            text=dataset.get_captions(), 
            text_encoder=get_text_embedding_model(text_encoder_name), 
            batch_size=batch_size, 
            model_variant=text_model_variant
            )
        upload_embeddings_to_hf(
            embeddings=text_embedding, 
            embeddings_saving_path=embeddings_saving_path, 
            hf_api=api, 
            repo_id=repo_id, 
            path_in_repo=f"{ds_name}_{text_encoder_name}_{text_model_variant}_text_embeddings.pkl"
        )