from buildnn.BuildNN import BuildNN


def main():
    obj = BuildNN(
        [ 
           'all-mpnet-base-v2', 
           'distiluse-base-multilingual-cased-v1',
           'distiluse-base-multilingual-cased-v2',
           'distiluse-base-multilingual-cased', 
           'distilbert-base-uncased', 
           'bert-base-multilingual-cased',  
        ],
                  device='cpu')
    obj.encode_from_xml("data_xml/")
    embeddings = obj.get_embeddings()
    print(embeddings)
    obj.save_embeddings_xml()


if __name__ == '__main__':
    main()