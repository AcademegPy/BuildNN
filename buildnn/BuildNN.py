import logging
import numpy
import pandas as pd
from typing import List, Mapping, Union, Iterable
from sentence_transformers import SentenceTransformer


class BuildNN:
    def __init__(self, model: Union[str, List] = 'distiluse-base-multilingual-cased'):
        self.model = model
        self.embeddings = None

    def encode(self, from_list: List[str]):
        if isinstance(self.model, str):
            model = SentenceTransformer(self.model)
            self.embeddings = {}
            embeddings = []
            for name in from_list:
                embeddings.append(numpy.array(model.encode(name), dtype='double'))
            self.embeddings[self.model] = embeddings
        
        elif isinstance(self.model, Iterable):
            self.embeddings = {}
            for model in self.model:
                sent_trans = SentenceTransformer(model)
                embeddings = []
                for name in from_list:
                    embeddings.append(numpy.array(sent_trans.encode(name), dtype='double'))
                    self.embeddings[model] = embeddings

        return self

    def get_embeddings(self, model: str = None) -> Union[dict, numpy.ndarray]:
        if len(self.embeddings) == 1:
            return self.embeddings[self.model]

        elif len(self.embeddings) > 1 and model:
            return self.embeddings[model]

        return self.embeddings






