from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from pymongo import MongoClient
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import random
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app) 

MONGO_URL = os.getenv('MONGO_URL')
client = MongoClient(MONGO_URL)
db = client['test']
cases = db['cases']
pacients = db['pacients']
users = db['users']

@dataclass
class Localidade:
    latitude: float = True
    longitude: float = True

@dataclass
class Users:
    nome: str = True
    email: str = True
    senha: str = True
    role: str = True

@dataclass
class Caso:
    titulo: str = True
    descricao: str = True
    status: str = True
    responsavel: Users = True
    dataAbertura: datetime = True
    dataFechamento: datetime = True
    dataOcorrencia: datetime = True
    localidade: Localidade 
    

@dataclass
class Paciente:
    peritoResponsavel: Users = True
    NIC: str = True
    nome: str
    genero: str = True
    idade: int
    documento: str
    endereco: str
    etnia: str
    odontograma: str
    regiaoAnatomicas: str

@app.route('/api/casos', methods=['GET'])
def listar_casos():
    documentos = list(cases.find({}, {"_id":0}))
    return jsonify(documentos), 200

@app.route('/api/pacientes', methods=['GET'])
def listar_pacientes():
    documentos = list(pacients.find({}, {"_id":0}))
    return jsonify(documentos), 200

@app.route('/api/usuarios', methods=['GET'])
def listar_usuarios():
    documentos = list(users.find({}, {"_id":0}))
    return jsonify(documentos), 200



if __name__ == '__main__':
    app.run(debug=True)
