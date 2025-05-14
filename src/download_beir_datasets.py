import os
from beir import util

BEIR_DATASETS = [
    "msmarco",  # MSMARCO
    "trec-covid",  # TREC-COVID
    "nfcorpus",  # NFCorpus
    "nq",  # NQ
    "hotpotqa",  # HotpotQA
    "fiqa",  # FiQA-2018
    "arguana",  # ArguAna
    "webis-touche2020",  # Touche-2020
    "cqadupstack",  # CQADupstack
    "quora",  # Quora
    "dbpedia-entity",  # DBPedia
    "scidocs",  # SCIDOCS
    "fever",  # FEVER
    "climate-fever",  # Climate-FEVER
    "scifact",  # SciFact
]

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
base_out_dir = os.path.join(project_root, "cache", "my_datasets")
os.makedirs(base_out_dir, exist_ok=True)

for dataset in BEIR_DATASETS:
    dataset_dir = os.path.join(base_out_dir, dataset)
    if os.path.exists(dataset_dir):
        print(
            f"Dataset '{dataset}' is already downloaded in {dataset_dir}. Skipping download."
        )
        continue

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    print(f"Downloading and unzipping '{dataset}' from {url} ...")

    data_path = util.download_and_unzip(url, base_out_dir)
    print(f"Dataset '{dataset}' has been downloaded and unzipped to: {data_path}\n")
