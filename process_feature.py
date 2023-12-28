import yaml
import argparse

from yaml.loader import FullLoader
from utils import process_feature

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', required=True)
    parser.add_argument('--load_model', required=True)
    args = parser.parse_args()
    yaml_path = args.yaml_path
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=FullLoader)
    load_model = args.load_model
    process_feature(config, load_model, process_subset=False)