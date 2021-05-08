from transformers import DistilBertTokenizer, DistilBertModel
from typing import List, Mapping, Union, Iterable
import torch
import numpy
from BuildNN import BuildNN


def main():
    obj = BuildNN(['distiluse-base-multilingual-cased', 'distilbert-base-uncased'])
    obj.encode("Hello, World!")
    embeddings = obj.get_embeddings('distilbert-base-uncased')
    print(embeddings)
    obj.save_embeddings_xml()


if __name__ == '__main__':
    main()