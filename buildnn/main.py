from transformers import DistilBertTokenizer, DistilBertModel
from typing import List, Mapping, Union, Iterable
import torch
import numpy
from BuildNN import BuildNN





def main():
    obj = BuildNN(['distiluse-base-multilingual-cased', 'distilbert-base-uncased'])
    obj.encode("Всем хай.")
    print(obj.get_embeddings())



if __name__ == '__main__':
    main()