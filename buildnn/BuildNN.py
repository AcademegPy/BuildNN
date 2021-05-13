import os
import numpy
import pandas as pd
import xml.etree.ElementTree as xml
from typing import List, Mapping, Union, Iterable
from sentence_transformers import SentenceTransformer


class BuildNN:
    def __init__(self, model: Union[str, List] = 'distiluse-base-multilingual-cased', device: str = 'cpu'):
        self.model = model
        self.device = device
        self.embeddings = None

    def encode(self, text: Union[str, List]):
        """ Embedding generation
        
        Parameters
        ----------
        text : str, list
            A list of texts or a single string instance
        """
        
        if isinstance(self.model, str):
            model = SentenceTransformer(self.model, device=self.device)
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
                sent_trans = SentenceTransformer(model, device=self.device)
                embeddings = []
                if isinstance(text, str):
                    embeddings.append(numpy.array(sent_trans.encode(text), dtype='double'))
                elif isinstance(text, list):
                    for token in text:
                      embeddings.append(numpy.array(sent_trans.encode(token), dtype='double'))
                self.embeddings[model] = embeddings

        return self


    def encode_from_xml(self, folder: str):
        """ Embedding generation for xml files
        
        Parameters
        __________
        folder : str
            Folder with documents in xml format
        """
        files = os.listdir(folder)
        self.embeddings = {}

        if isinstance(self.model, str):
            model = SentenceTransformer(self.model, device=self.device)
            embeddings = []
            for file in files:
                path = os.path.join(folder, file)
                parser = xml.XMLParser(encoding="utf-8")
                tree = xml.parse(path, parser=parser)
                text = tree.find('text').text
                embeddings.append(numpy.array(model.encode(text), dtype='double'))

            self.embeddings[self.model] = embeddings

        elif isinstance(self.model, Iterable):
            for model in self.model:
                sent_trans = SentenceTransformer(model, device=self.device)
                embeddings = []
                for file in files:
                    path = os.path.join(folder, file)
                    parser = xml.XMLParser(encoding="utf-8")
                    tree = xml.parse(path, parser=parser)
                    text = tree.find('text').text
                    embeddings.append(numpy.array(sent_trans.encode(text), dtype='double'))

                self.embeddings[model] = embeddings


    def get_embeddings(self, model: str = None) -> Union[dict, numpy.ndarray]:
        """ Getting embedding
        
        Parameters
        __________
        model : str
            Model name or path
        """
        
        if len(self.embeddings) == 1:
            return self.embeddings[self.model]

        elif len(self.embeddings) > 1 and model:
            return self.embeddings[model]

        return self.embeddings
    
    def save_embeddings_xml(self, model: str = None, filename: str = 'embeddings.xml'):
        """ Saving embeddings in xml format
        
        Parameters
        __________
        model : str
            Model name or path
        filename : str
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
        






