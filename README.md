PRÉDICTION DE MALADIE CARDIAQUE
-------------------------------------------------------------------------------
FONCTIONNALITÉS

L'application permet d'évaluer le risque de maladie cardiaque à partir de
11 paramètres cliniques standards.

FONCTIONNALITÉS PRINCIPALES :
-------------------------------------------------------------------------------
- PRÉDICTION EN TEMPS RÉEL
  Saisie des données patient et obtention immédiate du risque cardiaque
  (Risque élevé / Risque faible) avec probabilité associée.

- EXPLORATION DES DONNÉES
  Visualisation du jeu de données intégré (918 patients) avec options
  d'affichage (début, fin, échantillon aléatoire).

- VISUALISATIONS INTERACTIVES
  - Importance des variables médicales
  - Matrice de confusion
  - Courbe ROC
  - Métriques de classification
  - Distribution des cas

- DOUBLE MODÈLE
  - Random Forest : optimisation des performances
  - Régression Logistique : analyse des coefficients

- AIDES CONTEXTUELLES
  Descriptions médicales pour chaque paramètre clinique

- INTERFACE GRAPHIQUE MODERNE
  Application desktop avec thème sombre

TECHNOLOGIES UTILISÉES

LANGAGE
  - Python 3.8 ou supérieur

BIBLIOTHÈQUES PRINCIPALES
-------------------------------------------------------------------------------
  Catégorie Machine Learning :
  - scikit-learn 1.7.2 (Random Forest, Régression Logistique)
  
  Catégorie Traitement des données :
  - pandas (manipulation de données)
  - numpy (calculs numériques)
  
  Catégorie Visualisation :
  - matplotlib (graphiques)
  - seaborn (visualisations avancées)
  
  Catégorie Interface graphique :
  - customtkinter (interface moderne)

MODÈLES IMPLÉMENTÉS
-------------------------------------------------------------------------------
  Modèle 1 - RANDOM FOREST
  - Type : Ensemble de 100 arbres de décision
  - Usage : Production / Prédiction
  - Avantage : Meilleure performance (AUC-ROC 0.927)

  Modèle 2 - RÉGRESSION LOGISTIQUE
  - Type : Modèle linéaire
  - Usage : Analyse clinique / Interprétation
  - Avantage : Coefficients interprétables

FORMATS DE DONNÉES
  - CSV (jeu de données)
  - JSON (métriques)
  - Pickle (modèles sauvegardés)

CAS D'USAGE
-------------------------------------------------------------------------------
1 : DÉPISTAGE PRÉVENTIF
Contexte : Un patient consulte pour un bilan cardiaque de routine.
Utilisation : Le médecin saisit les paramètres du patient (âge, tension,
              cholestérol, résultats ECG) et obtient une évaluation
              instantanée du risque.
Bénéfice : Détection précoce des patients à risque.

2 : AIDE À LA DÉCISION CLINIQUE
Contexte : Patient présentant des symptômes évocateurs.
Utilisation : Le médecin utilise la prédiction comme outil d'aide avant de
              prescrire des examens approfondis (échographie, scintigraphie).
Bénéfice : Réduction des examens invasifs non nécessaires.

3 : ANALYSE ÉPIDÉMIOLOGIQUE
Contexte : Recherche sur les facteurs de risque cardiovasculaire.
Utilisation : Les chercheurs analysent l'importance des variables pour
              identifier les facteurs les plus prédictifs.
Bénéfice : Identification que ST_Slope, ChestPainType et ExerciseAngina
           sont les facteurs les plus importants.

4 : FORMATION MÉDICALE
Contexte : Enseignement aux étudiants en médecine.
Utilisation : Les étudiants explorent l'impact de chaque variable sur le
              risque cardiaque via l'interface interactive.
Bénéfice : Apprentissage concret des facteurs de risque cardiovasculaire.

5 : TÉLÉMÉDECINE
Contexte : Plateforme de téléconsultation.
Utilisation : Intégration du modèle pour un premier niveau de tri des
              patients avant consultation médicale.
Bénéfice : Priorisation des patients à risque élevé.

RÉSULTATS
-------------------------------------------------------------------------------

RANDOM FOREST - MODÈLE PRINCIPAL

Métriques sur l'ensemble de TEST :
  Accuracy (Exactitude)      : 0.875  (87.5% de prédictions correctes)
  Précision                  : 0.862  (86.2% des risques prédits sont réels)
  Recall (Sensibilité)       : 0.922  (92.2% des vrais risques détectés)
  F1-Score                   : 0.891  (Moyenne harmonique)
  AUC-ROC                    : 0.927  (Excellent pouvoir discriminant)
  Spécificité                : 0.817  (81.7% des non-malades corrects)

Métriques sur l'ensemble d'ENTRAÎNEMENT :
  Accuracy      : 0.911
  Précision     : 0.905
  Recall        : 0.938
  F1-Score      : 0.921
  AUC-ROC       : 0.980

INSTALLATION
================================================================================
PRÉREQUIS : Python 3.8+

Installer les dependences :
pip install pandas numpy scikit-learn matplotlib seaborn customtkinter

Entrainement du model :
python entrainement.py

Lancement :
python app.py





 

