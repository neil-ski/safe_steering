from extract_activations import extract_activations
from filtered_indices import indices

extract_activations(
    model_name = "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
    target_layers = [2, 4, 6, 8, 10, 12, 14, 15],
    pool_type = "finaltoken",
    batch_size= 8,
    max_len_input = 512,
    max_len_output = 128,
    checkpoint_period = 100,
    dataset= "LLM-LAT/harmful-dataset",
    seed=42,
    selected_indices=indices,
    # I checked the logs and realized my script crashed when ~95% done on the train dataset. 
    # I thought it finished and didn't restart it.
    # To replicate my results exactly you will need to cut off the train dataset at 1306. 
    train_cutoff=1306, 
)