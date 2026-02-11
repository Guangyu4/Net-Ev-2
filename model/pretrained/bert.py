from transformers import AutoTokenizer, AutoModel
import torch
import os

class BertEmbedding:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.cache_dir = os.path.join(os.path.dirname(__file__), "../../cache")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # os.environ['http_proxy'] = 'http://127.0.0.1:7890'
        # os.environ['https_proxy'] = 'http://127.0.0.1:7890'
        
        # Load tokenizer and model once during initialization
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = self.model.to(self.device)
        self.model.eval() 
    
    def get_embedding(self, text):
        if isinstance(text, tuple):
            text = list(text)
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings

_bert_embedding_instance = None

def GetEmbedding(text):
    global _bert_embedding_instance
    if _bert_embedding_instance is None:
        _bert_embedding_instance = BertEmbedding()
    return _bert_embedding_instance.get_embedding(text)

if __name__ == "__main__":
    bert_embedder = BertEmbedding()
    print(bert_embedder.get_embedding(('hello world','hello')).shape)
    # Test backward compatibility
    print(GetEmbedding(('hello world','hello')).shape)