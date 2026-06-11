import pickle
import json
import numpy as np
import pandas as pd
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
import sys
warnings.filterwarnings('ignore')

print("📂 Chargement du modèle de classification...")

# Charger les fichiers sauvegardés
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    feature_importance = pd.read_csv('feature_importance.csv')
    
    # Charger les données
    data = pd.read_csv('heart.csv')
    print("✓ Tous les fichiers chargés\n")
except FileNotFoundError as e:
    print(f"❌ Erreur de chargement: {e}")
    print("⚠️  Assurez-vous que les fichiers suivants sont présents:")
    print("   - model.pkl")
    print("   - scaler.pkl")
    print("   - feature_names.pkl")
    print("   - model_metrics.json")
    print("   - feature_importance.csv")
    print("   - heart.csv")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur inattendue: {e}")
    sys.exit(1)

# Hints détaillés pour chaque variable avec gestion dynamique
def get_hints_for_feature(feature):
    """Retourne les hints pour une feature donnée, avec valeurs par défaut"""
    hints_map = {
        'age': ('Âge (années)', '29 - 77', '55', 'Âge du patient'),
        'sex': ('Sexe', '0=Femme, 1=Homme', '1', 'Sexe du patient'),
        'cp': ('Type de douleur thoracique', '0 - 3', '1', '0=Typique, 1=Atypique, 2=Non angineuse, 3=Asymptomatique'),
        'trestbps': ('Pression artérielle (mmHg)', '90 - 200', '130', 'Au repos'),
        'chol': ('Cholestérol (mg/dl)', '125 - 564', '240', 'Taux de cholestérol'),
        'fbs': ('Glucose à jeun', '0 ou 1', '0', '0=<120mg/dl, 1=>120mg/dl'),
        'restecg': ('Résultat ECG au repos', '0 - 2', '1', 'Électrocardiogramme au repos'),
        'thalach': ('Fréquence cardiaque max', '60 - 202', '150', 'Battements par minute'),
        'exang': ('Angine induite', '0 ou 1', '0', '0=Non, 1=Oui'),
        'oldpeak': ('Dépression ST', '0.0 - 6.2', '1.0', 'Décalage du segment ST'),
        'slope': ('Pente du ST', '0 - 2', '1', 'Pente du segment ST à l\'exercice'),
        'ca': ('Vaisseaux calcifiés', '0 - 4', '0', 'Nombre de vaisseaux colorés'),
        'thal': ('Thalassémie', '0 - 3', '2', '0=Normal, 1=Défaut fixe, 2=Défaut réversible'),
    }
    
    # Si la feature existe dans le mapping, retourne ses valeurs
    if feature.lower() in hints_map:
        return hints_map[feature.lower()]
    
    # Sinon, génère des valeurs par défaut
    return (
        feature.capitalize(),  # Label
        '0 - 100',             # Range par défaut
        '0',                   # Valeur par défaut
        f'Valeur pour {feature}'  # Description
    )

# ==================== INTERFACE GRAPHIQUE ====================
class HeartDiseaseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("❤️ Prédiction Maladie Cardiaque - Classification")
        self.geometry("1700x950")
        
        try:
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
        except:
            pass
        
        self.resizable(True, True)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.input_vars = {}
        self.data = data  # Stocker les données chargées
        
        # Frame principal
        main_container = ctk.CTkFrame(self, fg_color="#0f1419")
        main_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        
        # Section gauche
        left_section = self.create_left_section(main_container)
        left_section.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        
        # Section droite
        right_section = self.create_right_section(main_container)
        right_section.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        
    def create_left_section(self, parent):
        """Crée la section gauche avec formulaire"""
        left_frame = ctk.CTkFrame(parent, fg_color="transparent")
        left_frame.grid_rowconfigure(2, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        # Header
        header = ctk.CTkFrame(left_frame, fg_color="#1a1f2e", corner_radius=15)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        title = ctk.CTkLabel(header, text="❤️ PRÉDICTION MALADIE CARDIAQUE", 
                            font=("Segoe UI", 26, "bold"), text_color="#e74c3c")
        title.pack(pady=15, padx=20)
        
        subtitle = ctk.CTkLabel(header, text="Classification - Risque de maladie cardiaque", 
                               font=("Segoe UI", 12), text_color="#888")
        subtitle.pack(pady=(0, 15), padx=20)
        
        # Formulaire scrollable
        form_frame = ctk.CTkFrame(left_frame, fg_color="#1a1f2e", corner_radius=15)
        form_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 20))
        form_frame.grid_rowconfigure(0, weight=1)
        form_frame.grid_columnconfigure(0, weight=1)
        
        scroll_frame = ctk.CTkScrollableFrame(form_frame, fg_color="transparent")
        scroll_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        scroll_frame.grid_columnconfigure(0, weight=1)
        scroll_frame.grid_columnconfigure(1, weight=1)
        
        row = 0
        for i, feature in enumerate(feature_names):
            col = i % 2
            if col == 0:
                row += 1
            
            # Utilisation de la fonction pour obtenir les hints
            label_text, range_text, default_val, hint_text = get_hints_for_feature(feature)
            
            input_container = ctk.CTkFrame(scroll_frame, fg_color="#242d3d", corner_radius=10)
            input_container.grid(row=row, column=col, sticky="ew", padx=8, pady=10)
            input_container.grid_columnconfigure(0, weight=1)
            
            label = ctk.CTkLabel(input_container, text=label_text, 
                               font=("Segoe UI", 11, "bold"), text_color="#e74c3c")
            label.pack(anchor="w", padx=12, pady=(10, 2))
            
            hint_label = ctk.CTkLabel(input_container, text=f"💡 {hint_text}", 
                                    font=("Segoe UI", 9), text_color="#00ff00")
            hint_label.pack(anchor="w", padx=12, pady=(0, 3))
            
            range_label = ctk.CTkLabel(input_container, text=f"Range: {range_text}  |  Défaut: {default_val}", 
                                    font=("Segoe UI", 9, "italic"), text_color="#666")
            range_label.pack(anchor="w", padx=12, pady=(0, 8))
            
            entry = ctk.CTkEntry(input_container, placeholder_text=default_val,
                               fg_color="#1a1f2e", text_color="#e74c3c",
                               border_color="#e74c3c", border_width=1,
                               font=("Segoe UI", 11))
            entry.pack(fill="x", padx=12, pady=(0, 10))
            entry.insert(0, default_val)
            self.input_vars[feature] = entry
        
        # Boutons
        button_container = ctk.CTkFrame(left_frame, fg_color="transparent")
        button_container.grid(row=3, column=0, sticky="ew", pady=(0, 20))
        button_container.grid_columnconfigure((0, 1, 2), weight=1)  # Ajout d'une 3ème colonne
        
        # Bouton pour explorer les données
        explore_btn = ctk.CTkButton(
            button_container, text="📊 EXPLORER DONNÉES", command=self.show_data_explorer,
            font=("Segoe UI", 12, "bold"), height=50,
            fg_color="#9b59b6", text_color="white",  # Couleur violette
            hover_color="#8e44ad", corner_radius=10
        )
        explore_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        
        predict_btn = ctk.CTkButton(
            button_container, text="🔮 PRÉDIRE", command=self.predict,
            font=("Segoe UI", 14, "bold"), height=50,
            fg_color="#e74c3c", text_color="white", 
            hover_color="#c0392b", corner_radius=10
        )
        predict_btn.grid(row=0, column=1, sticky="ew", padx=8)
        
        reset_btn = ctk.CTkButton(
            button_container, text="↻ RÉINITIALISER", command=self.reset_inputs,
            font=("Segoe UI", 13), height=50,
            fg_color="#1a1f2e", text_color="#888",
            border_color="#e74c3c", border_width=2,
            hover_color="#242d3d", corner_radius=10
        )
        reset_btn.grid(row=0, column=2, sticky="ew", padx=(8, 0))
        
        # Résultat
        result_container = ctk.CTkFrame(left_frame, fg_color="#1a1f2e", corner_radius=15)
        result_container.grid(row=4, column=0, sticky="ew", pady=(0, 20))
        
        self.result_label = ctk.CTkLabel(
            result_container, text="--",
            font=("Segoe UI", 36, "bold"), text_color="#e74c3c"
        )
        self.result_label.pack(pady=20)
        
        self.interpretation_label = ctk.CTkLabel(
            result_container, text="Entrez les valeurs et cliquez PRÉDIRE",
            font=("Segoe UI", 12), text_color="#888"
        )
        self.interpretation_label.pack(pady=(0, 20))
        
        # Métriques
        metrics_container = ctk.CTkFrame(left_frame, fg_color="#1a1f2e", corner_radius=15)
        metrics_container.grid(row=5, column=0, sticky="ew")
        metrics_container.grid_columnconfigure((0, 1, 2), weight=1)
        
        metrics_data = [
            ("Accuracy", f"{metrics.get('accuracy_test', 0):.3f}", "#00ff00"),
            ("AUC-ROC", f"{metrics.get('auc_test', 0):.3f}", "#00ff00"),
            ("F1-Score", f"{metrics.get('f1_test', 0):.3f}", "#00ff00")
        ]
        
        for i, (label, value, color) in enumerate(metrics_data):
            metric_box = ctk.CTkFrame(metrics_container, fg_color="#0f1419", corner_radius=10)
            metric_box.grid(row=0, column=i, sticky="ew", padx=8, pady=15)
            
            ctk.CTkLabel(metric_box, text=label, font=("Segoe UI", 11, "bold"), 
                        text_color="#888").pack(pady=(10, 3))
            ctk.CTkLabel(metric_box, text=value, font=("Segoe UI", 15, "bold"), 
                        text_color=color).pack(pady=(3, 10))
        
        return left_frame
    
    def create_right_section(self, parent):
        """Crée la section droite avec visualisations"""
        right_frame = ctk.CTkFrame(parent, fg_color="transparent")
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Header visualisations
        viz_header = ctk.CTkFrame(right_frame, fg_color="#1a1f2e", corner_radius=15)
        viz_header.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        viz_header.grid_columnconfigure(0, weight=1)
        
        title = ctk.CTkLabel(viz_header, text="📊 VISUALISATIONS ET ANALYSES", 
                            font=("Segoe UI", 18, "bold"), text_color="#e74c3c")
        title.pack(anchor="w", padx=20, pady=(12, 8))
        
        buttons_frame = ctk.CTkFrame(viz_header, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=20, pady=(0, 12))
        
        self.viz_var = ctk.StringVar(value="Importance")
        
        options = [
            "Importance",
            "Métriques",
            "Matrice Confusion",
            "ROC Curve",
            "Distribution"
        ]
        
        seg_btn = ctk.CTkSegmentedButton(
            buttons_frame, values=options, variable=self.viz_var,
            command=self.update_visualization,
            font=("Segoe UI", 11)
        )
        seg_btn.set("Importance")
        seg_btn.pack(side="left", fill="x", expand=True, padx=4)
        
        canvas_frame = ctk.CTkFrame(right_frame, fg_color="#1a1f2e", corner_radius=15)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.canvas_frame = canvas_frame
        self.update_visualization()
        
        return right_frame
    
    def show_data_explorer(self):
        """Affiche une fenêtre d'exploration des données"""
        if self.data is None or self.data.empty:
            self.show_error("Impossible de charger les données. Vérifiez que le fichier 'heart.csv' existe.")
            return
        
        # Créer une nouvelle fenêtre
        explorer_window = ctk.CTkToplevel(self)
        explorer_window.title("🔍 Exploration des Données - Heart Dataset")
        explorer_window.geometry("1300x750")
        explorer_window.resizable(True, True)
        
        # Configurer la grille
        explorer_window.grid_rowconfigure(0, weight=1)
        explorer_window.grid_columnconfigure(0, weight=1)
        
        # Frame principal
        main_frame = ctk.CTkFrame(explorer_window, fg_color="#0f1419")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Header
        header = ctk.CTkFrame(main_frame, fg_color="#1a1f2e", corner_radius=15)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        
        title = ctk.CTkLabel(header, text="📊 EXPLORATION DU DATASET HEART", 
                            font=("Segoe UI", 20, "bold"), text_color="#9b59b6")
        title.pack(pady=15, padx=20)
        
        # Informations générales
        info_frame = ctk.CTkFrame(header, fg_color="transparent")
        info_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        # Créer 4 colonnes pour les statistiques
        stats_columns = []
        for i in range(4):
            col_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
            col_frame.pack(side="left", expand=True, fill="x", padx=10)
            stats_columns.append(col_frame)
        
        # Ajouter les statistiques
        stats_info = [
            (f"Lignes: {self.data.shape[0]:,}", "Nombre total d'observations"),
            (f"Colonnes: {self.data.shape[1]}", "Nombre de variables"),
            (f"Valeurs: {self.data.size:,}", "Total des valeurs"),
            (f"Valeurs manquantes: {self.data.isnull().sum().sum()}", "Données incomplètes")
        ]
        
        for i, (value, description) in enumerate(stats_info):
            ctk.CTkLabel(stats_columns[i], text=value, 
                        font=("Segoe UI", 13, "bold"), text_color="#9b59b6").pack()
            ctk.CTkLabel(stats_columns[i], text=description, 
                        font=("Segoe UI", 9), text_color="#888").pack()
        
        # Contrôles d'affichage
        controls_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        controls_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        # Frame pour les options
        options_frame = ctk.CTkFrame(controls_frame, fg_color="#1a1f2e", corner_radius=10)
        options_frame.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(options_frame, text="Options d'affichage:", 
                     font=("Segoe UI", 12, "bold"), text_color="#9b59b6").pack(anchor="w", padx=15, pady=10)
        
        # Type d'affichage
        display_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
        display_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        self.display_type = ctk.StringVar(value="head")
        ctk.CTkRadioButton(display_frame, text="Début", variable=self.display_type, 
                          value="head", font=("Segoe UI", 11), text_color="#888").pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(display_frame, text="Fin", variable=self.display_type, 
                          value="tail", font=("Segoe UI", 11), text_color="#888").pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(display_frame, text="Aléatoire", variable=self.display_type, 
                          value="sample", font=("Segoe UI", 11), text_color="#888").pack(side="left")
        
        # Nombre de lignes
        rows_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
        rows_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        ctk.CTkLabel(rows_frame, text="Nombre de lignes:", 
                     font=("Segoe UI", 11), text_color="#888").pack(side="left")
        
        self.n_rows = ctk.CTkSlider(rows_frame, from_=5, to=100, number_of_steps=19, 
                                   command=lambda x: self.n_rows_label.configure(text=f"{int(float(x))}"))
        self.n_rows.set(15)
        self.n_rows.pack(side="left", fill="x", expand=True, padx=(10, 0))
        
        self.n_rows_label = ctk.CTkLabel(rows_frame, text="15", 
                                        font=("Segoe UI", 11, "bold"), text_color="#9b59b6")
        self.n_rows_label.pack(side="left", padx=(10, 0))
        
        # Bouton de rafraîchissement
        refresh_btn = ctk.CTkButton(controls_frame, text="🔄 Rafraîchir", 
                                   command=lambda: self.update_data_display(explorer_window),
                                   font=("Segoe UI", 12), height=40,
                                   fg_color="#9b59b6", text_color="white",
                                   hover_color="#8e44ad", corner_radius=10)
        refresh_btn.pack(side="right", fill="y")
        
        # Zone d'affichage des données
        data_display_frame = ctk.CTkFrame(main_frame, fg_color="#1a1f2e", corner_radius=15)
        data_display_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 15))
        data_display_frame.grid_rowconfigure(0, weight=1)
        data_display_frame.grid_columnconfigure(0, weight=1)
        
        # Scrollable frame pour les données
        scroll_frame = ctk.CTkScrollableFrame(data_display_frame, fg_color="transparent")
        scroll_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Stocker la référence pour mise à jour
        self.data_display_scroll = scroll_frame
        self.current_explorer_window = explorer_window
        
        # Mettre à jour l'affichage initial
        self.update_data_display(explorer_window)
        
        # Informations supplémentaires
        stats_frame = ctk.CTkFrame(main_frame, fg_color="#1a1f2e", corner_radius=15)
        stats_frame.grid(row=3, column=0, sticky="ew", pady=(0, 15))
        
        ctk.CTkLabel(stats_frame, text="📈 Informations sur les données:", 
                    font=("Segoe UI", 12, "bold"), text_color="#9b59b6").pack(anchor="w", padx=20, pady=10)
        
        # Informations détaillées
        details_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        details_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        # Colonnes supplémentaires
        detail_cols = []
        for i in range(3):
            col_frame = ctk.CTkFrame(details_frame, fg_color="transparent")
            col_frame.pack(side="left", expand=True, fill="x", padx=10)
            detail_cols.append(col_frame)
        
        # Types de données
        numeric_cols = len(self.data.select_dtypes(include=[np.number]).columns)
        object_cols = len(self.data.select_dtypes(include=['object']).columns)
        
        target_col = self.data.columns[-1]
        if self.data[target_col].dtype == 'object':
            target_values = self.data[target_col].unique()
        else:
            unique_vals = self.data[target_col].unique()
            target_values = [f"Valeur {v}" for v in unique_vals]
        
        detail_info = [
            (f"Colonnes numériques: {numeric_cols}", "Variables continues"),
            (f"Colonnes catégorielles: {object_cols}", "Variables discrètes"),
            (f"Cible: {target_col}", f"Classes: {', '.join(map(str, target_values))}")
        ]
        
        for i, (value, description) in enumerate(detail_info):
            ctk.CTkLabel(detail_cols[i], text=value, 
                        font=("Segoe UI", 11), text_color="#00ff00").pack()
            ctk.CTkLabel(detail_cols[i], text=description, 
                        font=("Segoe UI", 9), text_color="#888").pack()
        
        # Bouton de fermeture
        close_btn = ctk.CTkButton(main_frame, text="✕ Fermer", 
                                 command=explorer_window.destroy,
                                 font=("Segoe UI", 12), height=40,
                                 fg_color="#1a1f2e", text_color="#888",
                                 border_color="#9b59b6", border_width=2,
                                 hover_color="#242d3d", corner_radius=10)
        close_btn.grid(row=4, column=0, sticky="e", pady=(10, 0))
    
    def update_data_display(self, explorer_window=None):
        """Met à jour l'affichage des données dans la fenêtre d'exploration"""
        if not hasattr(self, 'data_display_scroll'):
            return
        
        # Nettoyer le contenu précédent
        for widget in self.data_display_scroll.winfo_children():
            widget.destroy()
        
        # Récupérer les paramètres
        n_rows = int(self.n_rows.get()) if hasattr(self, 'n_rows') else 15
        display_type = self.display_type.get() if hasattr(self, 'display_type') else "head"
        
        # Sélectionner les données selon l'option
        if display_type == "head":
            display_data = self.data.head(n_rows)
            title = f"PREMIÈRES {n_rows} LIGNES"
        elif display_type == "tail":
            display_data = self.data.tail(n_rows)
            title = f"DERNIÈRES {n_rows} LIGNES"
        else:  # sample
            display_data = self.data.sample(n=min(n_rows, len(self.data)))
            title = f"ÉCHANTILLON ALÉATOIRE DE {len(display_data)} LIGNES"
        
        # Titre
        title_label = ctk.CTkLabel(self.data_display_scroll, text=title,
                                  font=("Segoe UI", 14, "bold"), text_color="#00d4ff")
        title_label.pack(anchor="w", pady=(0, 15))
        
        # Créer un tableau
        table_frame = ctk.CTkFrame(self.data_display_scroll, fg_color="transparent")
        table_frame.pack(fill="x")
        
        # En-têtes du tableau
        headers_frame = ctk.CTkFrame(table_frame, fg_color="#242d3d")
        headers_frame.pack(fill="x", pady=(0, 2))
        
        # Afficher les en-têtes de colonnes
        for i, col in enumerate(display_data.columns):
            header_cell = ctk.CTkFrame(headers_frame, fg_color="#1a1f2e", width=100, height=35)
            header_cell.grid(row=0, column=i, sticky="nsew", padx=1, pady=1)
            header_cell.grid_propagate(False)
            
            # Ajuster la largeur des colonnes
            headers_frame.grid_columnconfigure(i, weight=1)
            
            ctk.CTkLabel(header_cell, text=str(col), 
                        font=("Segoe UI", 10, "bold"), text_color="#9b59b6").pack(expand=True, fill="both")
        
        # Afficher les données
        for row_idx, (_, row) in enumerate(display_data.iterrows()):
            row_color = "#1a1f2e" if row_idx % 2 == 0 else "#242d3d"
            row_frame = ctk.CTkFrame(table_frame, fg_color=row_color)
            row_frame.pack(fill="x", pady=1)
            
            for col_idx, (col_name, value) in enumerate(row.items()):
                cell_frame = ctk.CTkFrame(row_frame, fg_color="transparent", width=100, height=30)
                cell_frame.grid(row=0, column=col_idx, sticky="nsew", padx=1, pady=1)
                cell_frame.grid_propagate(False)
                
                # Formater la valeur
                if isinstance(value, float):
                    display_value = f"{value:.3f}" if abs(value) < 0.001 else f"{value:.2f}"
                else:
                    display_value = str(value)
                
                # Mettre en couleur la colonne cible
                text_color = "#ff4444" if col_name == self.data.columns[-1] and value == 1 else "#00ff00" if col_name == self.data.columns[-1] and value == 0 else "#888"
                
                ctk.CTkLabel(cell_frame, text=display_value,
                           font=("Segoe UI", 9),
                           text_color=text_color).pack(expand=True, fill="both")
                
                row_frame.grid_columnconfigure(col_idx, weight=1)
    
    def show_error(self, message):
        """Affiche un message d'erreur"""
        error_window = ctk.CTkToplevel(self)
        error_window.title("Erreur")
        error_window.geometry("400x150")
        
        ctk.CTkLabel(error_window, text="❌ Erreur", 
                    font=("Segoe UI", 16, "bold"), text_color="#ff4444").pack(pady=20)
        
        ctk.CTkLabel(error_window, text=message, 
                    font=("Segoe UI", 11), text_color="#888", wraplength=350).pack(padx=20)
        
        ctk.CTkButton(error_window, text="OK", command=error_window.destroy,
                     fg_color="#1a1f2e", text_color="#888").pack(pady=20)
    
    def predict(self):
        """Effectue une prédiction"""
        try:
            input_data = []
            for feature in feature_names:
                val_str = self.input_vars[feature].get().strip()
                if not val_str:
                    raise ValueError(f"Le champ '{feature}' est vide")
                val = float(val_str)
                input_data.append(val)
            
            input_array = np.array([input_data])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            disease_prob = probability[1]
            
            if prediction == 1:
                couleur = "#ff4444"
                status = "⚠️ RISQUE DÉTECTÉ"
                detail = f"Probabilité: {disease_prob*100:.1f}%"
            else:
                couleur = "#00ff00"
                status = "✓ RISQUE FAIBLE"
                detail = f"Probabilité: {(1-disease_prob)*100:.1f}%"
            
            self.result_label.configure(
                text=status,
                text_color=couleur
            )
            
            self.interpretation_label.configure(
                text=detail,
                text_color=couleur
            )
        
        except ValueError as e:
            self.result_label.configure(text="❌ ERREUR", text_color="#ff4444")
            self.interpretation_label.configure(
                text=f"Erreur: {str(e)}\nTous les champs doivent contenir des nombres valides",
                text_color="#ff4444"
            )
        except Exception as e:
            self.result_label.configure(text="❌ ERREUR", text_color="#ff4444")
            self.interpretation_label.configure(
                text=f"Erreur inattendue: {str(e)}",
                text_color="#ff4444"
            )
    
    def reset_inputs(self):
        """Réinitialise les champs"""
        for feature in feature_names:
            self.input_vars[feature].delete(0, "end")
            _, _, default_val, _ = get_hints_for_feature(feature)
            self.input_vars[feature].insert(0, default_val)
        self.result_label.configure(text="--", text_color="#e74c3c")
        self.interpretation_label.configure(
            text="Entrez les valeurs et cliquez PRÉDIRE",
            text_color="#888"
        )
    
    def update_visualization(self, value=None):
        """Met à jour la visualisation"""
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        viz_type = self.viz_var.get()
        
        viz_container = ctk.CTkFrame(self.canvas_frame, fg_color="transparent")
        viz_container.pack(fill="both", expand=True)
        viz_container.grid_rowconfigure(0, weight=1)
        viz_container.grid_columnconfigure(0, weight=1)
        
        canvas_frame = ctk.CTkFrame(viz_container, fg_color="#1a1f2e", corner_radius=15)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        fig = Figure(figsize=(9, 5.5), dpi=90, facecolor='#1a1f2e')
        
        explanation = ""
        
        try:
            if viz_type == "Importance":
                ax = fig.add_subplot(111)
                ax.set_facecolor('#0f1419')
                ax.tick_params(colors='#888', labelsize=10)
                for spine in ax.spines.values():
                    spine.set_color('#2a3f5f')
                self.plot_importance(ax)
                ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
                explanation = " IMPORTANCE DES VARIABLES\n\n" \
                             "Montre les 10 facteurs les plus importants pour la prédiction du risque cardiaque.\n" \
                             "La fréquence cardiaque et le type de douleur thoracique sont souvent clés.\n" \
                             "Les variables en haut influencent le plus la décision du modèle."
                
            elif viz_type == "Métriques":
                ax = fig.add_subplot(111)
                ax.set_facecolor('#0f1419')
                ax.tick_params(colors='#888', labelsize=10)
                for spine in ax.spines.values():
                    spine.set_color('#2a3f5f')
                self.plot_metrics(ax)
                ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
                explanation = "MÉTRIQUES DE CLASSIFICATION\n\n" \
                             "Accuracy: Pourcentage de prédictions correctes.\n" \
                             "Precision: % des risques prédits qui sont réellement présents.\n" \
                             "Recall: % des vrais risques correctement détectés.\n" \
                             "F1-Score: Moyenne harmonique de Precision et Recall."
                
            elif viz_type == "Matrice Confusion":
                ax = fig.add_subplot(111)
                ax.set_facecolor('#0f1419')
                self.plot_confusion_matrix(ax)
                explanation = "🔲 MATRICE DE CONFUSION\n\n" \
                             "Montre les 4 types de résultats:\n" \
                             "• Vrais Négatifs (TN): Pas risque, prédit pas risque ✓\n" \
                             "• Faux Positifs (FP): Pas risque, prédit risque ✗\n" \
                             "• Faux Négatifs (FN): Risque, prédit pas risque ✗ (critique!)\n" \
                             "• Vrais Positifs (TP): Risque, prédit risque ✓"
                
            elif viz_type == "ROC Curve":
                ax = fig.add_subplot(111)
                ax.set_facecolor('#0f1419')
                ax.tick_params(colors='#888', labelsize=10)
                for spine in ax.spines.values():
                    spine.set_color('#2a3f5f')
                self.plot_roc_curve(ax)
                ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
                explanation = "COURBE ROC (Receiver Operating Characteristic)\n\n" \
                             "Montre le compromis entre détection correcte et faux positifs.\n" \
                             "Plus la courbe est proche du coin supérieur gauche, meilleur est le modèle.\n" \
                             f"AUC-ROC: {metrics.get('auc_test', 0):.4f} (plus haut = mieux, max=1.0)"
                
            elif viz_type == "Distribution":
                ax = fig.add_subplot(111)
                ax.set_facecolor('#0f1419')
                ax.tick_params(colors='#888', labelsize=10)
                for spine in ax.spines.values():
                    spine.set_color('#2a3f5f')
                self.plot_distribution(ax)
                ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
                explanation = " DISTRIBUTION DES CAS\n\n" \
                             "Montre la proportion des cas avec et sans risque cardiaque.\n" \
                             "Un déséquilibre important peut affecter les performances du modèle.\n" \
                             "Les techniques de rééquilibrage peuvent être utilisées si nécessaire."
        
        except Exception as e:
            print(f"Erreur visualisation ({viz_type}): {e}")
            ax = fig.add_subplot(111)
            ax.set_facecolor('#0f1419')
            ax.text(0.5, 0.5, f"Erreur de chargement\n{viz_type}\n\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes,
                   color='#ff4444', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        explanation_frame = ctk.CTkFrame(viz_container, fg_color="#1a1f2e", corner_radius=15)
        explanation_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        explanation_frame.grid_columnconfigure(0, weight=1)
        
        explanation_label = ctk.CTkLabel(
            explanation_frame, text=explanation,
            font=("Segoe UI", 11), text_color="#e74c3c", wraplength=1000, justify="left"
        )
        explanation_label.pack(padx=15, pady=12)
    
    def plot_importance(self, ax):
        top = feature_importance.head(10)
        y_pos = np.arange(len(top))
        ax.barh(y_pos, top['importance'].values, color='#e74c3c', edgecolor='white', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top['feature'].values, fontsize=11, color='#e74c3c', fontweight='bold')
        ax.set_xlabel('Importance', color='#888', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 Variables Impactantes', color='#e74c3c', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
    
    def plot_metrics(self, ax):
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        values = [
            metrics.get('accuracy_test', 0),
            metrics.get('precision_test', 0),
            metrics.get('recall_test', 0),
            metrics.get('f1_test', 0),
            metrics.get('auc_test', 0)
        ]
        colors = ['#e74c3c' if v < 0.8 else '#f39c12' if v < 0.9 else '#00ff00' for v in values]
        ax.bar(range(len(metrics_names)), values, color=colors, edgecolor='white', linewidth=1)
        ax.set_xticks(range(len(metrics_names)))
        ax.set_xticklabels(metrics_names, color='#888', fontsize=10, rotation=15)
        ax.set_ylabel('Score', color='#888', fontsize=12, fontweight='bold')
        ax.set_title('Métriques de Classification', color='#e74c3c', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', color='white', fontweight='bold')
    
    def plot_confusion_matrix(self, ax):
        df = pd.read_csv('heart.csv')
        target_col = df.columns[-1]
        if df[target_col].dtype == 'object':
            df[target_col] = (df[target_col].isin(['Yes', 'yes', '1'])).astype(int)
        elif df[target_col].max() > 1:
            df[target_col] = (df[target_col] > 0).astype(int)
        
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != target_col:
                df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
        X = df_encoded.drop(target_col, axis=1)
        y = df_encoded[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        im = ax.imshow(cm, cmap='Reds', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pas Maladie', 'Maladie'], color='white')
        ax.set_yticklabels(['Pas Maladie', 'Maladie'], color='white')
        ax.set_xlabel('Prédiction', color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel('Réalité', color='white', fontsize=12, fontweight='bold')
        ax.set_title('Matrice de Confusion', color='#e74c3c', fontsize=13, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=20, fontweight='bold')
    
    def plot_roc_curve(self, ax):
        df = pd.read_csv('heart.csv')
        target_col = df.columns[-1]
        if df[target_col].dtype == 'object':
            df[target_col] = (df[target_col].isin(['Yes', 'yes', '1'])).astype(int)
        elif df[target_col].max() > 1:
            df[target_col] = (df[target_col] > 0).astype(int)
        
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != target_col:
                df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
        X = df_encoded.drop(target_col, axis=1)
        y = df_encoded[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test_scaled = scaler.transform(X_test)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        ax.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'ROC (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Aléatoire (AUC = 0.5)')
        ax.set_xlabel('Taux Faux Positifs', color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel('Taux Vrais Positifs', color='white', fontsize=12, fontweight='bold')
        ax.set_title('Courbe ROC', color='#e74c3c', fontsize=13, fontweight='bold')
        ax.legend(facecolor='#1a1f2e', edgecolor='white', labelcolor='white', fontsize=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
    
    def plot_distribution(self, ax):
        df = pd.read_csv('heart.csv')
        target_col = df.columns[-1]
        if df[target_col].dtype == 'object':
            labels = df[target_col].unique().tolist()
            values = [len(df[df[target_col] == l]) for l in labels]
        else:
            values = [(df[target_col] == 0).sum(), (df[target_col] == 1).sum()]
            labels = ['Pas Maladie', 'Maladie']
        
        colors = ['#00ff00', '#e74c3c']
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                           colors=colors, textprops={'color': 'white', 'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Distribution des Cas', color='#e74c3c', fontsize=13, fontweight='bold')

if __name__ == "__main__":
    try:
        print("✅ Lancement de l'application...\n")
        app = HeartDiseaseApp()
        app.mainloop()
    except Exception as e:
        print(f" Erreur lors du lancement: {e}")
        import traceback
        traceback.print_exc()