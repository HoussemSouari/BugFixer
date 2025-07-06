import yaml
import os

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Set default values
    config.setdefault("fp16", True)
    config.setdefault("gradient_accumulation_steps", 1)
    config.setdefault("logging_steps", 100)
    config.setdefault("eval_steps", 500)
    config.setdefault("save_steps", 1000)
    
    # Handle Kaggle environment
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        config["output_dir"] = "/kaggle/working/output"
        config["fp16"] = True
    
    return config