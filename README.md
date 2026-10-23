# 🎓 Academic Admission Prediction — Licence d'Excellence (PFE Master IA)

> **Projet de Fin d'Études (PFE) — Master d'Excellence en Intelligence Artificielle**  
> *Faculté des Sciences Ben M'Sick (FSBM), Université Hassan II de Casablanca*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-red.svg)](https://streamlit.io/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced%20Data-SMOTE%20%7C%20ADASYN-green.svg)](https://imbalanced-learn.org/)
[![CatBoost](https://img.shields.io/badge/Algorithm-CatBoost%20%7C%20LightGBM-orange.svg)](https://catboost.ai/)

---

## 📌 Présentation du Projet

Ce projet de fin d'études développe un **système d'aide à la décision algorithmique** pour assister la commission pédagogique lors de la sélection des candidats à une **Licence d'Excellence**.

À partir de données académiques réelles (parcours du baccalauréat, moyenne du diplôme, progression semestrielle S1 à S4, sessions de rattrapage), le système prédit la probabilité d'admission d'un étudiant et identifie les dossiers à fort potentiel.

---

## 🔬 Défi Technique : Le Déséquilibre Sévère de Classes (Imbalanced Data)

Dans les concours sélectifs, seule une petite minorité de candidats est admise ($\approx 5\text{--}15\%$). L'optimisation classique sur l'accuracy mène à un modèle trivial incapable de détecter les bons candidats.

### Méthodologies Implémentées :
1. **Échantillonnage Avancé** :
   - `SMOTE` (*Synthetic Minority Over-sampling Technique*)
   - `ADASYN` (*Adaptive Synthetic Sampling*)
   - Sous-échantillonnage contrôlé de la classe majoritaire
2. **Métriques Adaptées** : Optimisation sur le **F1-score**, la précision et le rappel de la classe minoritaire (Admis).
3. **Partitionnement Stratifié** : Découpage `Train / Validation / Test` stratifié garantissant la représentativité des classes à chaque étape.

---

## 🚀 Lancer l'Application Streamlit

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/pfe-admission-app.git
cd pfe-admission-app

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application web
streamlit run app.py
```

### Fonctionnalités de l'Interface :
- **👤 Évaluation Individuelle** : Saisie du parcours étudiant (Bac, semestres, rattrapages) avec verdict instantané et facteurs explicatifs.
- **📂 Évaluation par Lot** : Import d'un tableau Excel/CSV complet de candidats, calcul des probabilités, classement automatisé et export des résultats.

---

## 📁 Structure du Répertoire

```
pfe-admission-app/
├── app.py                      # Application Web Streamlit (Portail Commission)
├── projet_pfe.ipynb            # Notebook complet de modélisation & SMOTE/ADASYN
├── requirements.txt            # Dépendances Python
└── README.md                   # Documentation du projet
```
