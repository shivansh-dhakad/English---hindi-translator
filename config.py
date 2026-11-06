from pathlib import Path


def get_config():
    return {
        # data
        "data_file": "Dataset_English_Hindi.csv",
        "lang_src": "English",
        "lang_tgt": "Hindi",
        "seq_len": 128,            

        # model
        "d_model": 256,
        "N": 6,                    # encoder/decoder layers
        "h": 8,                    
        "d_ff": 1024,
        "dropout": 0.1,

        # training 
        "batch_size": 64,          
        "num_epoch": 20,
        "lr": 3e-4,                
        "warmup_steps": 1000,
        "label_smoothing": 0.1,
        "grad_clip": 1.0,
        "num_workers": 2,
        "seed": 42,

        # files 
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",      
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = Path(config['model_folder'])
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(model_folder / model_filename)


def latest_weights_file_path(config):
    files = sorted(Path(config['model_folder']).glob(f"{config['model_basename']}*.pt"))
    return str(files[-1]) if files else None
