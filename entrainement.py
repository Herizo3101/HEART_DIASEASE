"""
╔════════════════════════════════════════════════════════════════════════════╗
║              ENTRAÎNEMENT CLASSIFICATION - HEART DISEASE                   ║
║                     Prédiction de maladie cardiaque                        ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("🔧 ENTRAÎNEMENT CLASSIFICATION - MALADIE CARDIAQUE")
print("="*80)

# ==================== ÉTAPE 1: CHARGEMENT DES DONNÉES ====================
try:
    df = pd.read_csv('heart.csv')
    print(f"✓ Fichier chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"✓ Variables: {list(df.columns)}\n")
except FileNotFoundError:
    print("❌ Erreur: Le fichier 'heart.csv' n'existe pas!")
    exit()

# ==================== ÉTAPE 2: ANALYSE EXPLORATOIRE ====================
print ("en tete ")
print(df.head())
print("📊 Statistiques descriptives:")
print(df.describe())

print("\n📈 Distribution de la cible (Maladie cardiaque):")
target_col = df.columns[-1]  # Dernière colonne généralement
print(df[target_col].value_counts())

# ==================== ÉTAPE 3: CORRÉLATIONS ====================
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
print("\n🔗 CORRÉLATIONS AVEC MALADIE CARDIAQUE:")
print(correlations.head(10))

# ==================== ÉTAPE 4: PRÉPARATION DES DONNÉES ====================
df_encoded = df.copy()

# Encodage de la cible si nécessaire
if df_encoded[target_col].dtype == 'object':
    df_encoded[target_col] = (df_encoded[target_col].isin(['Yes', 'yes', '1'])).astype(int)
elif df_encoded[target_col].max() > 1:
    df_encoded[target_col] = (df_encoded[target_col] > 0).astype(int)

# Encodage des variables catégoriques
categorical_cols = df_encoded.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != target_col:
        df_encoded[col] = pd.factorize(df_encoded[col])[0]

X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

print(f"\n✓ X shape: {X.shape} | y shape: {y.shape}")
print(f"✓ Distribution: Pas maladie: {(y==0).sum()} | Maladie: {(y==1).sum()}")

# ==================== ÉTAPE 5: NORMALISATION ====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Données normalisées - Train: {X_train_scaled.shape[0]} | Test: {X_test_scaled.shape[0]}")

print("\n" + "="*80)
print("🤖 ENTRAÎNEMENT DU MODÈLE - RANDOM FOREST CLASSIFIER")
print("="*80)

# ==================== ÉTAPE 6: ENTRAÎNEMENT ====================
print("\n⏳ Entraînement du modèle de classification...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("✓ Modèle entraîné")

# ==================== ÉTAPE 7: VALIDATION CROISÉE ====================
print("\n📊 Validation croisée 5-Fold...")
scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

cv_results = cross_validate(model, X_train_scaled, y_train, cv=5, scoring=scoring_metrics)

cv_accuracy = cv_results['test_accuracy']
cv_precision = cv_results['test_precision']
cv_recall = cv_results['test_recall']
cv_f1 = cv_results['test_f1']

print("\n✓ Résultats de la validation croisée:")
print(f"  Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
print(f"  Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
print(f"  Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
print(f"  F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# ==================== ÉTAPE 8: PRÉDICTIONS ET MÉTRIQUES ====================
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)
y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]

# Métriques train
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train)
recall_train = recall_score(y_train, y_pred_train)
f1_train = f1_score(y_train, y_pred_train)
auc_train = roc_auc_score(y_train, y_pred_proba_train)

# Métriques test
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
auc_test = roc_auc_score(y_test, y_pred_proba_test)

# ==================== ÉTAPE 9: MATRICE DE CONFUSION ====================
cm_test = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm_test.ravel()
specificity_test = tn / (tn + fp)
sensitivity_test = recall_test

print("\n" + "="*80)
print("📈 RÉSULTATS FINAUX")
print("="*80)

print(f"\n{'Métrique':<25} {'Train':<15} {'Test':<15} {'Écart':<12}")
print("-"*80)
print(f"{'Accuracy':<25} {accuracy_train:.4f}         {accuracy_test:.4f}         {abs(accuracy_train-accuracy_test):.4f}")
print(f"{'Precision':<25} {precision_train:.4f}         {precision_test:.4f}         {abs(precision_train-precision_test):.4f}")
print(f"{'Recall (Sensibilité)':<25} {recall_train:.4f}         {recall_test:.4f}         {abs(recall_train-recall_test):.4f}")
print(f"{'Spécificité':<25} {'--':<15} {specificity_test:.4f}")
print(f"{'F1-Score':<25} {f1_train:.4f}         {f1_test:.4f}         {abs(f1_train-f1_test):.4f}")
print(f"{'AUC-ROC':<25} {auc_train:.4f}         {auc_test:.4f}         {abs(auc_train-auc_test):.4f}")
print("-"*80)

# Matrice de confusion
print(f"\n📊 MATRICE DE CONFUSION (Test Set):")
print(f"   Vrais Négatifs (TN):  {tn}")
print(f"   Faux Positifs (FP):   {fp}")
print(f"   Faux Négatifs (FN):   {fn}")
print(f"   Vrais Positifs (TP):  {tp}")

# ==================== ÉTAPE 10: IMPORTANCE DES VARIABLES ====================
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n⭐ TOP 10 VARIABLES LES PLUS IMPACTANTES:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:<20} {row['importance']:.4f}")

# ==================== ÉTAPE 11: VISUALISATION ====================
print("\n📊 Génération des visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Métriques
ax = axes[0, 0]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_vals = [accuracy_train, precision_train, recall_train, f1_train]
test_vals = [accuracy_test, precision_test, recall_test, f1_test]
x = np.arange(len(metrics_names))
width = 0.35
ax.bar(x - width/2, train_vals, width, label='Train', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, test_vals, width, label='Test', color='#3498db', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, rotation=15)
ax.set_ylabel('Score')
ax.set_title('Métriques de Classification')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Importance
ax = axes[0, 1]
top_features = feature_importance.head(10)
ax.barh(range(len(top_features)), top_features['importance'].values, color='#e74c3c')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values, fontsize=9)
ax.set_xlabel('Importance')
ax.set_title('Top 10 Variables Impactantes')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Plot 3: Matrice de Confusion
ax = axes[1, 0]
im = ax.imshow(cm_test, cmap='Blues', aspect='auto')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Pas Maladie', 'Maladie'])
ax.set_yticklabels(['Pas Maladie', 'Maladie'])
ax.set_xlabel('Prédiction')
ax.set_ylabel('Réalité')
ax.set_title('Matrice de Confusion')
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm_test[i, j], ha="center", va="center", color="w", fontsize=20, fontweight='bold')

# Plot 4: ROC Curve
ax = axes[1, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC Curve (AUC = {auc_test:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Aléatoire')
ax.set_xlabel('Taux de Faux Positifs')
ax.set_ylabel('Taux de Vrais Positifs')
ax.set_title('Courbe ROC')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heart_disease_analysis.png', dpi=100, bbox_inches='tight')
print("✓ Visualisations sauvegardées: heart_disease_analysis.png")
plt.close()

print("\n" + "="*80)
print("💾 SAUVEGARDE DU MODÈLE")
print("="*80)

# Sauvegarde
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ model.pkl")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ scaler.pkl")

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("   ✓ feature_names.pkl")

metrics_data = {
    'accuracy_test': float(accuracy_test),
    'accuracy_train': float(accuracy_train),
    'precision_test': float(precision_test),
    'precision_train': float(precision_train),
    'recall_test': float(recall_test),
    'recall_train': float(recall_train),
    'f1_test': float(f1_test),
    'f1_train': float(f1_train),
    'auc_test': float(auc_test),
    'auc_train': float(auc_train),
    'specificity_test': float(specificity_test),
    'cv_accuracy_mean': float(cv_accuracy.mean()),
    'cv_accuracy_std': float(cv_accuracy.std()),
    'cv_f1_mean': float(cv_f1.mean()),
    'cv_f1_std': float(cv_f1.std()),
    'tn': int(tn),
    'fp': int(fp),
    'fn': int(fn),
    'tp': int(tp),
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=4)
print("   ✓ model_metrics.json")

feature_importance.to_csv('feature_importance.csv', index=False)
print("   ✓ feature_importance.csv")

print("\n" + "="*80)
print("✅ ENTRAÎNEMENT TERMINÉ - MODÈLE CLASSIFICATION PRÊT")
print("="*80)
print("\nLance maintenant: python app.py")
print("="*80 + "\n")