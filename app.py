from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# Carregar modelo treinado (pipeline + label encoder)
with open("modelo_treinado_recife.pkl", "rb") as f:
    model_data = pickle.load(f)
    pipeline = model_data["pipeline"]
    label_encoder = model_data["label_encoder"]


@app.route("/api/predict_nic", methods=["POST"])
def prever_nic():
    try:
        dados = request.json

        # Verificar se os campos obrigatórios estão presentes
        obrigatorios = ["genero", "idade", "etnia", "endereco"]
        if not all(campo in dados for campo in obrigatorios):
            return jsonify({"erro": "Campos obrigatórios ausentes"}), 400

        # Criar DataFrame de entrada
        X_input = pd.DataFrame(
            [
                {
                    "genero": dados["genero"],
                    "idade": dados["idade"],
                    "etnia": dados["etnia"],
                    "endereco": dados["endereco"],
                }
            ]
        )

        # Fazer predição
        y_pred = pipeline.predict(X_input)
        y_label = label_encoder.inverse_transform(y_pred)

        return jsonify({"NIC_previsto": y_label[0]}), 200

    except Exception as e:
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
