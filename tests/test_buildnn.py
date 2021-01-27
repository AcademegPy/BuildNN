import sys
sys.path.append("./")
from buildnn.BuildNN import BuildNN

# Выбирается базовая модель distiluse-base-multilingual-cased
def test_base():
    obj = BuildNN().encode('Вот—вот должно было взойти солнце.')
    print(obj.get_embeddings())

# test_base()

# Задаем конкретную модель
def test_model(model):
    obj = BuildNN(model).encode('Вот—вот должно было взойти солнце.')
    print(obj.get_embeddings())

# test_model('distiluse-base-multilingual-cased')

# Список моделей с выбором эмеддингов конкретной модели
def test_model_list(model_list, model):
    obj = BuildNN(model_list).encode('Вот—вот должно было взойти солнце.')
    print(obj.get_embeddings(model))

test_model_list(['distiluse-base-multilingual-cased', 'distilbert-base-uncased'], 'distiluse-base-multilingual-cased')


