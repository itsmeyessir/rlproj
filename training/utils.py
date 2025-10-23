import json

def load_config(file_path):
    """Loads a JSON configuration file."""
    with open(file_path, 'r') as f:
        return json.load(f)