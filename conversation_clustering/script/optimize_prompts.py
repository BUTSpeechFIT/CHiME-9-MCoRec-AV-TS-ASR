import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from optimization.optimize_prompts import optimize_pairwise_prompts
from clustering_datasets import Chime9

if __name__ == "__main__":
    dataset = Chime9()
    optimize_pairwise_prompts(dataset, "fill_in_your_model", "api_url")
