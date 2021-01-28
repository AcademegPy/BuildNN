import numpy
import pytest
import sys
sys.path.append("./")
from buildnn.BuildNN import BuildNN
from tests.utils import get_test_strings

from_list = get_test_strings()

# Выбирается базовая модель distiluse-base-multilingual-cased
def test_base():
    obj = BuildNN().encode(from_list)
    embeddings = obj.get_embeddings()

    assert isinstance(embeddings, list)
    assert len(embeddings) == 6


# Задаем конкретную модель
@pytest.mark.parametrize("model", ['distiluse-base-multilingual-cased'])
def test_model(model):
    obj = BuildNN(model).encode(from_list)
    embeddings = obj.get_embeddings()

    assert isinstance(embeddings, list)
    assert len(embeddings) == 6


# Список моделей с выбором эмеддингов конкретной модели
@pytest.mark.parametrize("model_list, model", [(['distiluse-base-multilingual-cased', 'distilbert-base-uncased'],'distiluse-base-multilingual-cased')])
def test_model_list(model_list, model):
    obj = BuildNN(model_list).encode(from_list)
    model_embeddings = obj.get_embeddings(model)
    embeddings = obj.get_embeddings()

    assert isinstance(model_embeddings, list)
    assert isinstance(embeddings, dict)
    assert len(model_embeddings) == 6
    assert len(embeddings) == 2


