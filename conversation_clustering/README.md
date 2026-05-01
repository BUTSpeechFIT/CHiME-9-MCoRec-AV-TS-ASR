# Multi-conversation LLM-based clustering

This repository contains part of CHiME-9 Task 1 - MCoRec solution, implementing speaker clustering using LLMs and timestamp information.


## System description
The solution was implemented using the **DSPy** library. Multiple clustering strategies were developed and empirically evaluated, resulting in the following modules:

- **cluster.JointCluster**  
  Performs speaker clustering for an entire session in a single LLM invocation, jointly reasoning over all speakers and their transcripts to produce a global clustering.

- **cluster.PairwiseCluster**  
  Estimates topic-based similarity scores for every pair of speakers within a session. These scores are subsequently aggregated using **agglomerative clustering** to derive the final speaker groups.

- **cluster.PairwiseWithRefinement**  
  Extends the pairwise clustering approach by applying a modified joint clustering step as a post-processing phase, allowing the model to refine and correct the initial cluster assignments.

- **cluster.HybridCluster**  
  Integrates semantic pairwise topic similarity with temporal overlap cues. This method is specifically designed for cases where the transcript content alone does not provide sufficient semantic signal, leveraging timing information to improve clustering robustness.

## Installation

Run following to install dependencies. It is advised to use virtual environment such as `mamba`:

```
mamba create -n mcorec-cluster python==3.13
mamba activate mcorec-cluster

pip install -r requirements.txt
pip install -e .
```

## Usage

Export your LLM API provider API key

```
export API_KEY=<your_api_key>
```

### Running inference
More info at: https://dspy.ai/learn/evaluation/overview/

You can use a prepared script, which will run inference on MCoRec evalset and save the output.

```
python script/inference.py
```

For more info, run

```
python script/inference.py -h
```

### Running evaluation experiments
Each experiment is defined as `.yaml` configuration file in `experiments/`. General settings for LLM (API base, model, mlflow settings) are located in `config/config.yaml`.

To run all experiments, use:

```
bash script/run_all_exps.sh
```

To run a particular experiment, use:
```
python script/run_evaluation experiments/<path_to_yaml_conf>
```

All experiments are logged into a mlflow database. To access them, you can run a local UI server:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db -p 5000
```

#### Exporting clustering outputs from MLFlow experiments

A script is provided for exporting outputs from mlflow experiments. After obtaining the evaluation run ID from mlflow, run following:

```
python script/extract_outputs_from_mlflow.py <run_id>
```

---

### Additional notes
- Evaluation is multi-threaded, resulting in many API calls at once. It might be needed to increase open files limit (`ulitmit -n`) or decrease the thread count in experiment config files.
