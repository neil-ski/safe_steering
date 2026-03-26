import os
from typing import Dict, List, Optional, Tuple
from google.cloud import storage
import numpy as np
from numpy.lib.format import open_memmap
import torch
from util_types import DataSplit, ModelName, PoolingStrategy

# you must put your GCP credentials in this file 
CREDENTIALS_PATH: str = "gcs_key.json"
GOOGLE_APPLICATION_CREDENTIALS: str = "GOOGLE_APPLICATION_CREDENTIALS"

# this the name of your bucket
BUCKET_NAME: str = "activation_probe_data"

ROOT_DIR    = "/data/scratch/activation_probes"
os.makedirs(ROOT_DIR, exist_ok=True)

# this exposes your GCP service account credentials to the application
# https://docs.cloud.google.com/docs/authentication/application-default-credentials 
if GOOGLE_APPLICATION_CREDENTIALS not in os.environ:
    os.environ[GOOGLE_APPLICATION_CREDENTIALS] = CREDENTIALS_PATH

# save a global client and bucket to reuse TCP connection
CLIENT = storage.Client()
BUCKET = CLIENT.bucket(BUCKET_NAME)

def get_x_filename(model_name: str, layer_no: int, pool_type: str, split: str) -> str:
    return f"{model_name.split('/')[-1]}-L{layer_no}-{pool_type}-{split}-X.npy"

def get_y_filename(model_name: str, pool_type: str, split: str) -> str:
    return f"{model_name.split('/')[-1]}-{pool_type}-{split}-y.npy"

def get_checkpoint_filename(model_name: str, pool_type: str, split: str) -> str:
    return f"{model_name.split('/')[-1]}-{pool_type}-checkpoint_{split}.txt"

def get_rng_state_filename(model_name: str, pool_type: str, split: str) -> str:
    return f"{model_name.split('/')[-1]}-{pool_type}-rng_state_{split}.pt"

def get_local_path(prefix: str, file_name: str) -> str:
    if prefix and len(prefix) > 0:
        full_dir = os.path.join(ROOT_DIR, prefix)
        os.makedirs(full_dir, exist_ok=True)
        return os.path.join(full_dir, file_name)
    return os.path.join(ROOT_DIR, file_name)

def get_gcs_blob_name(prefix: str, file_name: str) -> str:
    if prefix and len(prefix) > 0:
        return f"{prefix}/{file_name}"
    
    return file_name

def upload_checkpoint(
        split: DataSplit, 
        idx: int, 
        target_layers: List[int], 
        pool_type: PoolingStrategy,
        model_name: ModelName,
        rng_state: Dict,
        prefix: str,
):
    print(f"\nUploading checkpoint for {split} at idx {idx} to GCS...")
    try:

        # upload the input X values
        for layer_no in target_layers:
            x_file_name = get_x_filename(model_name, layer_no, pool_type, split)
            local_path = get_local_path(prefix=prefix, file_name=x_file_name)
            blob = BUCKET.blob(get_gcs_blob_name(prefix=prefix, file_name=x_file_name))
            blob.upload_from_filename(local_path)
        
        # upload the target y values
        y_file_name = get_y_filename(model_name, pool_type, split)
        local_path = get_local_path(prefix, y_file_name)
        blob = BUCKET.blob(get_gcs_blob_name(prefix, y_file_name))
        blob.upload_from_filename(local_path)
        
        # record the checkpoint number
        # We reuse the same file name so that if we crash and come back,
        # we can read from the same file and determine what was the last number data value we were on.
        chkpt_file_name = get_checkpoint_filename(model_name, pool_type, split)
        chkpt_blob = BUCKET.blob(get_gcs_blob_name(prefix, chkpt_file_name))
        chkpt_blob.upload_from_string(str(idx))
        
        # record the RNG sequence state 
        rng_file_name = get_rng_state_filename(model_name, pool_type, split)
        torch.save(rng_state, get_local_path(prefix, rng_file_name))
        rng_blob = BUCKET.blob(get_gcs_blob_name(prefix, rng_file_name))
        rng_blob.upload_from_filename(get_local_path(prefix, rng_file_name))

        print("Checkpoint uploaded successfully.")
    except Exception as e:
        print(f"Error uploading to GCS: {e}")

def resume_checkpoint_helper(split: DataSplit, target_layers: List[int], pool_type: PoolingStrategy, model_name: ModelName, prefix: str):
    try:
        # get checkpoint number
        chkpt_file_name = get_checkpoint_filename(model_name, pool_type, split)
        chkpt_blob = BUCKET.blob(get_gcs_blob_name(
            prefix=prefix, 
            file_name=chkpt_file_name
        ))

        if not chkpt_blob.exists():
            print(f"No checkpoint found.")
            return 0
            
        idx = int(chkpt_blob.download_as_text())
        print(f"Found checkpoint for {split} at idx {idx}. Syncing files from GCS...")
        
        # load the saved input X files
        for layer_no in target_layers:
            x_file_name = get_x_filename(model_name, layer_no, pool_type, split)
            local_path = get_local_path(prefix, x_file_name)
            blob = BUCKET.blob(get_gcs_blob_name(prefix=prefix, file_name=x_file_name))
            if blob.exists() and not os.path.exists(local_path):
                print(f"Downloading {blob.name} to {local_path}...")
                blob.download_to_filename(local_path)

        # load the saved target y file
        y_file_name = get_y_filename(model_name, pool_type, split)
        local_path = get_local_path(prefix=prefix, file_name=y_file_name)
        blob = BUCKET.blob(get_gcs_blob_name(prefix=prefix, file_name=y_file_name))
        if blob.exists() and not os.path.exists(local_path):
            print(f"Downloading {blob.name} to {local_path}...")
            blob.download_to_filename(local_path)
            
        # load the RNG state file
        rng_file_name = get_rng_state_filename(model_name, pool_type, split)
        local_path = get_local_path(prefix, rng_file_name)
        blob = BUCKET.blob(get_gcs_blob_name(prefix=prefix, file_name=rng_file_name))
        if blob.exists() and not os.path.exists(local_path):
            print(f"Downloading {blob.name} to {local_path}...")
            blob.download_to_filename(local_path)
            
        return idx
    except Exception as e:
        print(f"Error checking GCS checkpoint: {e}")
        return 0

def resume_checkpoint_if_exists(
        split: DataSplit, 
        target_layers: List[int], 
        pool_type: PoolingStrategy, 
        model_name: ModelName, 
        prefix: str,
        n: Optional[int] = None, 
        d: Optional[int] = None,
    ) -> Tuple[int, dict[int, np.memmap], np.memmap, Optional[Dict]]:
    checkpoint_idx: int = resume_checkpoint_helper(
        split=split, 
        target_layers=target_layers, 
        pool_type=pool_type, 
        model_name=model_name, 
        prefix=prefix
    )
    # If we have a checkpoint append to the end of the file. Else create a new 
    # file if a file with the name doesn't exist or overwrite if it does.
    mode = "r+" if checkpoint_idx > 0 else "w+"
    if mode == "w+" and (n is None or d is None):
        raise ValueError("n and d must be provided to create a new memmap file.")

    # when we resumed the checkpoint above, it loaded the checkpointed files into local files 
    # dict from layer number to memory-mapped np array of activation at that layer for each sample in dataset
    x_arr_dict: dict[int, np.memmap] = {
        layer_no: open_memmap(
               get_local_path(prefix=prefix, file_name=get_x_filename(model_name, layer_no, pool_type, split)),
               mode=mode, dtype="float32", shape=(n, d) if mode == "w+" else None
           )
        for layer_no in target_layers
    }

    # memory-mapped np array of safe/unsafe class for each sample in dataset
    y_arr: np.memmap = open_memmap(
        get_local_path(prefix=prefix, file_name=get_y_filename(model_name, pool_type, split)),       
        mode=mode, 
        dtype="int64", 
        shape=(n,) if mode == "w+" else None
    )
    
    # load the RNG sequence state if it was downloaded
    rng_state = None
    rng_local_path = get_local_path(prefix=prefix, file_name=get_rng_state_filename(model_name, pool_type, split))
    if os.path.exists(rng_local_path):
        rng_state = torch.load(rng_local_path, weights_only=False)
        
    return checkpoint_idx, x_arr_dict, y_arr, rng_state