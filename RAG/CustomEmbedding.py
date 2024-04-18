from chromadb.api.types import EmbeddingFunction,Documents,Embeddings
from chromadb.api.types import EmbeddingFunction,Documents,Embeddings

from typing import Any, Dict, List

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    # Since we do dynamic imports we have to type this as Any
    models: Dict[str, Any] = {}

    # If you have a beefier machine, try "gtr-t5-large".
    # for a full list of options: https://huggingface.co/sentence-transformers, https://www.sbert.net/docs/pretrained_models.html
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "mps",
        normalize_embeddings: bool = False,
    ):
        if model_name not in self.models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ValueError(
                    "The sentence_transformers python package is not installed. Please install it with `pip install sentence_transformers`"
                )
            self.models[model_name] = SentenceTransformer(model_name, device=device)
        self._model = self.models[model_name]
        self._normalize_embeddings = normalize_embeddings

    def __call__(self, texts: Documents) -> Embeddings:
        return self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        ).tolist()
    
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        ).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings =self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        )
        #print(embeddings)
        return embeddings.tolist()
    

class Text2VecEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese",
                 device: str = "mps"):
        try:
            from text2vec import SentenceModel
        except ImportError:
            raise ValueError(
                "The text2vec python package is not installed. Please install it with `pip install text2vec`"
            )
        self._model = SentenceModel(model_name_or_path=model_name,device=device)

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), convert_to_numpy=True).tolist()  # type: ignore # noqa E501


    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._model.encode(text, convert_to_numpy=True).tolist() # type: ignore # noqa E501
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = self._model.encode(list(texts), convert_to_numpy=True)
        #print(embeddings)
        return embeddings.tolist() # type: ignore # noqa E501
