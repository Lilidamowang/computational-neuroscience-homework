import os
import json
import yaml
import argparse


class Option:

    def __init__(self, config_path):
        self.config_path = config_path
        self.config = yaml.load(open(config_path))
