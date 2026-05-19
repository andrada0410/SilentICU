import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Modelele disponibile pentru comparare
MODELE = {
    "minilm": "all-MiniLM-L6-v2", # mic, rapid
    "mpnet":  "all-mpnet-base-v2", # mai mare, mai precis
}


# Embedder -> transformare mesaje de alarma in vectori numerici
# ideea de baza:
#   "Heart rate too high" → [0.12, -0.45, 0.87, ...]  (384 numere)
#   "Tachycardia detected" → [0.11, -0.43, 0.85, ...]  (384 numere)
# Vectori apropiati in spatiu -> alarme similare semantic ( calculan prin cosine similarity)
# AlarmEmbedder -> transforma mesajele de alarma in vectori semantici
# embedder = AlarmEmbedder(model_name="minilm")
# vectori = embedder.embed(["Heart rate too high", "Tachycardia detected"])
# sim = embedder.similarity_matrix()  # matrice NxN de similaritati
class AlarmEmbedder:
    def __init__(self, model_name: str = "minilm"):
        model_id = MODELE.get(model_name, model_name)
        print(f"Inacracare model: {model_id} ...")
        self.model = SentenceTransformer(model_id)
        self.model_name = model_name
        self.embeddings = None # vectorii calculati
        self.messages = None # mesajele originale

    # transformare lista de mesaje in vectori
    # param:messages - lista de mesaje
    # return: matrice numpy de tipul (n_alarme, 384)
    def embed(self, messages: list) -> np.ndarray:
        self.messages = list(messages)
        print(f"Calculare embeddings pentru {len(messages)} alarme...")

        self.embeddings = self.model.encode(
            self.messages,
            show_progress_bar=True,
            normalize_embeddings=True,  # important: normalizare pentru cosine similarity
            batch_size=64,
        )
        print(f"Shape embeddings: {self.embeddings.shape}")
        return self.embeddings

    # calculare similitudine cosinus intre toti vectorii
    # return: matrice NxN unde [i][j] = similitudinea intre alarma i și j
    #         ( valori intre 0 (complet diferite) si 1 (identice) )
    def similarity_matrix(self) -> np.ndarray:
        if self.embeddings is None:
            raise ValueError("Ruleaza embed() mai intai!")
        return cosine_similarity(self.embeddings)

    # gasire cele mai similare alarme pentru mesaj nou
    # param: query - mesajul de cautat
    #      : cate rezultate sa returneze
    # returns: lista cu cele mai similare alarme si scorurile lor
    def find_similar(self, query: str, top_k: int = 5) -> list:
        if self.embeddings is None:
            raise ValueError("Ruleaza embed() mai intai!")

        query_vec = self.model.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [
            {
                "index": int(i),
                "message":  self.messages[i],
                "similarity": round(float(sims[i]), 4),
            }
            for i in top_indices
        ]

    # salveaza embeddings pe disc
    def save(self, path: str = "data/embeddings/embeddings.npy"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.embeddings)
        print(f"Embeddings salvate: {self.embeddings.shape} → {path}")

    # incarca embedding-urile salvate
    def load(self, path: str = "data/embeddings/embeddings.npy"):
        self.embeddings = np.load(path)
        print(f"Embeddings încărcate: {self.embeddings.shape}")
        return self.embeddings