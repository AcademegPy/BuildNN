from transformers import DistilBertTokenizer, DistilBertModel
from typing import List, Mapping, Union, Iterable
import torch
import numpy
from buildnn.BuildNN import BuildNN


def main():
    obj = BuildNN(['distiluse-base-multilingual-cased', 'distilbert-base-uncased'])
    obj.encode("Hello, World!")
    embeddings = obj.get_embeddings()
    print(embeddings)
    obj.save_embeddings_xml()


if __name__ == '__main__':
    main()