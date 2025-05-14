# contextual-relevance

# Running the Experiments
To run BRIGHT or BEIR experiments, navigate to the respective directoryin the eval folder and run the scripts in the order: retrieval.py, reranking.py, eval_metrics.py.

Example:

```
cd /src/eval/BRIGHT
python retrieval.py      # 1. Retrieve candidate documents using BM25s as your first-stage retriever
python reranking.py      # 2. Re-rank candidates using a reranking algorithm
python eval_metrics.py   # 3. Compute metrics
```

# Training the Setwise Reranking Model

We use axolotl (https://github.com/axolotl-ai-cloud/axolotl) to train our setwise reranking model. The file containing our rule-based rewards is 'setwise_grpo.py' and the config script for starting rl post-training in axolotl is 'Setwise.yaml'.