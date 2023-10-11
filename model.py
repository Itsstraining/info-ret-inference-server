import os
import urllib.request
import zipfile
import json

# from os.path import abspath, joins

import torch
import pandas as pd

from utils.data import BSARDataset
from utils.eval import BiEncoderEvaluator
from models.trainable_dense_models import BiEncoder
from sentence_transformers import util

# define class StoredModel


class StoredModel:
    # constructor
    def __init__(
        self,
        model_path,
        full_text_path,
        model,
        document_ids,
        d_embeddings,
        device,
        documents,
        batch_size=2,
    ):
        self.model_path = model_path
        self.full_text_path = full_text_path
        self.model = model
        self.document_ids = document_ids
        self.d_embeddings = d_embeddings
        self.device = device
        self.documents = documents
        self.batch_size = batch_size

    def infer(self, prompt, k=10):
        q_embeddings = self.model.q_encoder.encode(
            texts=[prompt], device=self.device, batch_size=self.batch_size
        )
        all_results = util.semantic_search(
            query_embeddings=q_embeddings,
            corpus_embeddings=self.d_embeddings,
            top_k=k,
            score_function=util.dot_score,
        )
        all_results = [
            [result["corpus_id"] for result in results] for results in all_results
        ]
        results = []
        for result in all_results[0]:
            print(result)
            results.append(
                {
                    "id": result,
                    "data": self.documents[result],
                }
            )
        return results


def fetch_model(url):
    # Download the model
    model_filename = url.split("/")[-1]
    urllib.request.urlretrieve(url, model_filename)

    # Unzip the model to the specified folder
    with zipfile.ZipFile(model_filename, "r") as zip_ref:
        zip_ref.extractall("./assets/models")

    # Remove the downloaded zip file
    os.remove(model_filename)


def cap_string(text, cap=200):
    if len(text) > cap:
        return text[:cap] + "..."
    else:
        return text


def load_model(model_path, full_text_path, batch_size=2):
    model = BiEncoder.load(model_path)
    documents_df = pd.read_csv(full_text_path)
    documents_dict = {}
    # for each row in documents_df
    for idx, row in documents_df.iterrows():
        # content, title, cat_1, cat_2, book_no, page_no
        doc_dict = {
            "id": row["id"],
            "content": cap_string(row["content"]),
            "title": row["title"],
            "cat1": row["cat_1"],
            "cat2": row["cat_2"],
            "book_no": row["book_no"],
            "page_no": row["page_no"],
        }
        # add the dictionary to the list
        documents_dict[row["id"]] = doc_dict

    id_doc_pair = documents_df.set_index("id")["content"].to_dict()
    document_ids = list(id_doc_pair.keys())
    documents = [id_doc_pair[doc_id] for doc_id in document_ids]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 2

    d_embeddings = model.d_encoder.encode(
        texts=documents, device=device, batch_size=batch_size
    )
    return StoredModel(
        model_path,
        full_text_path,
        model,
        document_ids,
        d_embeddings,
        device,
        documents_dict,
        batch_size,
    )
