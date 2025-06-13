import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("dados_ficticios_recife.csv")
df.dropna(inplace=True)

X = df[["genero", "idade", "etnia", "endereco"]]
Y = df["NIC"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)

categorical_features = ["genero", "etnia", "endereco"]
numeric_features = ["idade"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"))
])

pipeline.fit(X, y_encoded)

joblib.dump({"pipeline": pipeline, "label_encoder": label_encoder}, "modelo_treinado_recife.joblib")

print("Modelo treinado e salvo como 'modelo_treinado_recife.joblib'")
