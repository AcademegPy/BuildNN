import logging
import numpy
import pandas as pd
import xml.etree.ElementTree as xml
from typing import List, Mapping, Union, Iterable
from sentence_transformers import SentenceTransformer


class BuildNN:
    def __init__(self, model: Union[str, List] = 'distiluse-base-multilingual-cased'):
        self.model = model
        self.embeddings = None

    def encode(self, text: Union[str, List]):
        """ Генерация эмбеддингов
        
            from_list:  str - строка
                        list - список строк
        """
        
        if isinstance(self.model, str):
            model = SentenceTransformer(self.model)
            self.embeddings = {}
            embeddings = []
            if isinstance(text, str):
                embeddings.append(numpy.array(model.encode(text), dtype='double'))
            elif isinstance(text, list):
                for token in text:
                  embeddings.append(numpy.array(model.encode(token), dtype='double'))
            self.embeddings[self.model] = embeddings
        
        elif isinstance(self.model, Iterable):
            self.embeddings = {}
            for model in self.model:
                sent_trans = SentenceTransformer(model)
                embeddings = []
                if isinstance(text, str):
                    embeddings.append(numpy.array(sent_trans.encode(text), dtype='double'))
                elif isinstance(text, list):
                    for token in text:
                      embeddings.append(numpy.array(sent_trans.encode(token), dtype='double'))
                self.embeddings[model] = embeddings

        return self

    def get_embeddings(self, model: str = None) -> Union[dict, numpy.ndarray]:
        """ Получение эмбеддингов
        
            model: str - название модели
        """

        if len(self.embeddings) == 1:
            return self.embeddings[self.model]

        elif len(self.embeddings) > 1 and model:
            return self.embeddings[model]

        return self.embeddings
    
    def save_embeddings_xml(self, model: str = None, filename: str = 'embeddings.xml'):
        """ Сохранение эмбеддингов в формате XML
        
            model: str - название модели
            filename: str - имя файла
        """

        data = xml.Element("data")

        if model:
            embeddings_tag = xml.SubElement(data, "embeddings", name=model)
            embeddings = self.get_embeddings(model)
            for embedding in embeddings:
                xml.SubElement(embeddings_tag, "vec").text = str(embedding)

        elif len(self.model) > 1:
            for model in self.model:
                embeddings_tag = xml.SubElement(data, "embeddings", name=model)
                embeddings = self.get_embeddings(model)
                for embedding in embeddings:
                    xml.SubElement(embeddings_tag, "vec").text = str(embedding)

        tree = xml.ElementTree(data)
        tree.write(filename)
        






