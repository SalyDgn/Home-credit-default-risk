import pandas as pd
import pickle
from flask import Flask, request, jsonify
import os
import sys

# Ajoute le répertoire parent du répertoire de votre script au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from settings.params import MODEL_PARAMS


app = Flask(__name__)

# Charger le modèle MLflow

# Configurer l'URI de suivi pour utiliser ngrok
# mlflow.set_tracking_uri("https://4fb0-35-202-19-40.ngrok-free.app/")

# logged_model = 'runs:/4236c31fb5354cdfb6d11e60d4c9abcf/LGBMClassifier'

# Load model as a PyFuncModel.
# model = mlflow.pyfunc.load_model(logged_model)

# Charger le modèle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# Charger les données de test (préalablement lues)
test_data = pd.read_csv('test_data_final.csv', index_col=0)


# Filtrer les features selon feature_selection
feature_selection = MODEL_PARAMS["features_selected"] + ["SK_ID_CURR"]


@app.route('/')
def home():
    return "Hello, World!"


test_data_filtered = test_data[feature_selection]

# Définir le seuil de décision pour accorder un prêt
threshold = 0.5


# Route pour la prédiction
@app.route('/predict', methods=['GET'])
def predict():
    client_id = request.args.get('client_id')

    # Extraire les données du client
    client_data = test_data_filtered[test_data_filtered['SK_ID_CURR'] == int(client_id)]

    if client_data.empty:
        return jsonify({'error': 'Client ID not found'}), 404

    # Retirer l'ID avant la prédiction (si le modèle ne l'utilise pas)
    client_data = client_data.drop(columns=['SK_ID_CURR'])

    # Prédiction
    probability = model.predict_proba(client_data)[:, 1][0]  # Probabilité de défaut de paiement

    # Générer une phrase descriptive de la probabilité
    prob_statement = f"La probabilité que le client soit en défaut de paiement est de {probability:.2%}."

    # Conclusion basée sur le seuil
    if probability >= threshold:
        recommendation = (
            "Il est recommandé de ne pas accorder le prêt en raison d'un risque élevé de défaut de paiement."
        )
    else:
        recommendation = "Il est recommandé d'accorder le prêt, car le risque de défaut de paiement est faible."

    # Retourner la probabilité et la recommandation
    return jsonify(
        {'client_id': client_id, 'default_probability_statement': prob_statement, 'recommendation': recommendation}
    )


port = int(os.environ.get("PORT", 5000))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
