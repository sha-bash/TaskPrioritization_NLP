import json
import yaml


def read_json(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
        return config

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        return config