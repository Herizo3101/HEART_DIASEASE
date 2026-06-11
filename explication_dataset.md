1. Age – Âge du patient (en années)
2. Sex – Sexe du patient
   · M : Masculin
   · F : Féminin
3. ChestPainType – Type de douleur thoracique
   · ATA : Angine typique
   · NAP : Douleur non angineuse
   · ASY : Asymptomatique
   · TA : Angine atypique
4. RestingBP – Pression artérielle au repos (en mm Hg)
5. Cholesterol – Taux de cholestérol sérique (en mg/dl)
6. FastingBS – Glycémie à jeun
   · 1 : > 120 mg/dl
   · 0 : ≤ 120 mg/dl
7. RestingECG – Résultats électrocardiographiques au repos
   · Normal : Normal
   · ST : Anomalie de l’onde ST-T
   · LVH : Hypertrophie ventriculaire gauche probable ou certaine
8. MaxHR – Fréquence cardiaque maximale atteinte (battements par minute)
9. ExerciseAngina – Angine induite par l’exercice
   · Y : Oui
   · N : Non
10. Oldpeak – Dépression du segment ST induite par l’exercice par rapport au repos (en unités)
11. ST_Slope – Pente du segment ST à l’effort maximal
    · Up : Ascendante
    · Flat : Plate
    · Down : Descendante
12. HeartDisease – Variable cible (présence de maladie cardiaque)
    · 1 : Maladie cardiaque
    · 0 : Absence de maladie cardiaque



EXPLICATION DE NOM DE VARIABLE DANS LE RESULTATS


Voici la signification médicale de chaque feature dans le contexte de la prédiction des maladies cardiaques :

ST_Slope (Pente du segment ST)

· Signification : Pente du segment ST sur l'électrocardiogramme (ECG) pendant l'effort
· Échelle : Normalement "up-sloping" (montante), "flat" (plate), ou "down-sloping" (descendante)
· Importance : Une pente descendante ("down-sloping") indique une ischémie myocardique

ExerciseAngina (Angine à l'effort)

· Signification : Présence de douleur thoracique pendant l'exercice physique
· Type : Variable binaire (Oui/Non)
· Signe clinique : Symptôme typique de maladie coronarienne

ChestPainType (Type de douleur thoracique)

· Signification : Caractéristiques de la douleur thoracique
· Types courants :
  · TA : Angine typique
  · ATA : Angine atypique
  · NAP : Douleur non angineuse
  · ASY : Asymptomatique

FastingBS (Glycémie à jeun)

· Signification : Taux de sucre dans le sang après 8h de jeûne
· Seuil clinique : ≥ 120 mg/dL (6.7 mmol/L) considéré comme diabétique
· Risque : Le diabète augmente le risque cardiovasculaire

Oldpeak (Dépression ST)

· Signification : Dépression du segment ST induite par l'exercice par rapport au repos
· Mesure : En millimètres (mm) ou millivolts (mV) sur l'ECG
· Interprétation : Valeur plus élevée = plus d'ischémie myocardique

RestingECG (ECG au repos)

· Signification : Résultats de l'électrocardiogramme au repos
· Catégories :
  · Normal : Pas d'anomalies
  · ST : Anomalies de l'onde ST-T
  · LVH : Hypertrophie ventriculaire gauche

RestingBP (Pression artérielle au repos)

· Signification : Pression artérielle systolique au repos (en mmHg)
· Valeurs normales : < 120 mmHg (optimale)
· Hypertension : ≥ 130 mmHg

Age (Âge)

· Signification : Âge du patient en années
· Facteur de risque : Le risque cardiovasculaire augmente avec l'âge

MaxHR (Fréquence cardiaque maximale)

· Signification : Fréquence cardiaque maximale atteinte pendant l'effort (en bpm)
· Formule théorique : 220 - âge
· Interprétation : Capacité cardiaque à s'adapter à l'effort

Cholesterol (Cholestérol)

· Signification : Taux de cholestérol sérique en mg/dL
· Types : Généralement le cholestérol total
· Risque : Niveau élevé = risque accru d'athérosclérose

Sex (Sexe)

· Signification : Sexe biologique du patient
· Codage typique : 0 = Femme, 1 = Homme
· Risque : Les hommes ont généralement un risque plus élevé à âge égal

📌 Contexte Clinique

Ces features sont standard dans les scores de risque cardiovasculaire comme le score de Framingham. Elles permettent d'évaluer la probabilité de maladie cardiaque en combinant :

· Symptômes (douleur thoracique, angine)
· Tests diagnostiques (ECG, cholestérol, glycémie)
· Facteurs de risque traditionnels (âge, sexe, pression artérielle)

Chaque feature contribue à une évaluation globale du risque cardiovasculaire du patient.