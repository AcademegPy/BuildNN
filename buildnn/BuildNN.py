import logging
import numpy
import pandas as pd
from typing import List, Mapping, Union, Iterable
from sentence_transformers import SentenceTransformer


class BuildNN:
    def __init__(self, model: Union[str, List] = 'distiluse-base-multilingual-cased'):
        self.model = model
        self.embeddings = None

    def encode(self, text):
        if isinstance(self.model, str):
            model = SentenceTransformer(self.model)
            self.embeddings = {}
            self.embeddings[self.model] = numpy.array(model.encode(text), dtype='double')
        
        elif isinstance(self.model, Iterable):
            self.embeddings = {}
            for model in self.model:
                sent_trans = SentenceTransformer(model)
                self.embeddings[model] = numpy.array(sent_trans.encode(text), dtype='double')

        return self

    def get_embeddings(self, model: str = None) -> Union[dict, numpy.ndarray]:
        if len(self.embeddings) == 1:
            return self.embeddings[self.model]

        elif len(self.embeddings) > 1 and model:
            return self.embeddings[model]

        return self.embeddings






