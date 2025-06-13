import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# 1. Carregar os dados do CSV
df = pd.read_csv("dados_ficticios_recife.csv")

# 2. Remover registros com valores nulos
df.dropna(inplace=True)

# 3. Separar variáveis explicativas (X) e alvo (Y)
X = df[["genero", "idade", "etnia", "endereco"]]
Y = df["NIC"]

# 4. Codificar a variável alvo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)

# 5. Pré-processamento com OneHotEncoder para variáveis categóricas
categorical_features = ["genero", "etnia", "endereco"]
numeric_features = ["idade"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# 6. Criar pipeline com XGBoost
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"))
])

# 7. Treinar o modelo
pipeline.fit(X, y_encoded)

# 8. Salvar o pipeline e o label encoder
with open("modelo_treinado_recife.pkl", "wb") as f:
    pickle.dump({
        "pipeline": pipeline,
        "label_encoder": label_encoder
    }, f)

print(" Modelo treinado e salvo como 'modelo_treinado_recife.pkl'")
