import pandas as pd
from pymongo import MongoClient
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

# 1. Conectar no MongoDB e puxar dados
MONGO_URL = os.getenv('MONGO_URL')
client = MongoClient(MONGO_URL)
db = client['test']
colecao = db['pacients']

dados = list(colecao.find({}, {"_id": 0}))

# 2. Preparar DataFrame Flat

lista = []
for d in dados:
    lista.append({
        "genero": d["genero"],
        "idade": d["idade"],
        "etnia": d["etnia"],
        "endereco": d["endereco"],
        "NIC": d["NIC"]
    })

df = pd.DataFrame(lista)

# 3. Variáveis explicativas e alvo

X= df[["genero", "idade", "etnia", "endereco"]]
Y = df["NIC"]


# 4. Encode da variável alvo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)

# 5. Pipeline
categorical_features = ["genero", "etnia", "endereco"]
numeric_features = ["idade"]

preprocessor = ColumnTransformer(
    transformers= [
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(use_label_encoder=False, eval_metric = "mlogloss"))

])

# 6. Treinar
pipeline.fit(X, y_encoded)

# 7. Salvar pipeline + label encoder

with open("model.pkl", "wb") as f:
    pickle.dump({
        "pipeline": pipeline,
        "label_encoder": label_encoder
    }, f)

print("Modelo treinado e salvo em model.pkl")