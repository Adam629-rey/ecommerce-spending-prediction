# Démarrer le projet

1. Extraire l'archive et ouvrir un terminal dans `ecommerce-spending-prediction`.
2. Installer Python 3.13 si nécessaire.
3. Créer un environnement : `python -m venv .venv`.
4. Sous PowerShell, l'activer : `.venv\Scripts\Activate.ps1`.
5. Installer : `python -m pip install -r requirements.txt`.
6. Lancer : `python -m streamlit run streamlit_app.py`.

Pour régénérer les résultats : `python -m spending.train`.
Pour tester : installer `requirements-dev.txt`, puis exécuter `python -m pytest -q`.

Le README en anglais détaille la méthode, les limites, la publication GitHub et le déploiement Streamlit. Les rapports ont été calculés sur les données fournies : R² test 0,9890, RMSE 8,934 et MAE 7,228.

Les colonnes Email, Address et Avatar ont été retirées du fichier livré. Les 500 observations et les cinq colonnes numériques sont conservées. Le script initial a été remplacé par les modules `spending/model.py` et `spending/train.py`.

Avant de rendre les données publiques, vérifier les conditions de redistribution du cours. Leur licence n'était pas incluse dans les fichiers reçus. Aucune publication ni aucun déploiement n'a été effectué en votre nom.
