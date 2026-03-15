# src/data_processing.py


# ===============================
# === IMPORTS PRINCIPAUX ========
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# =====================================================
# === LISTES DES COLONNES NUMÉRIQUES & CATÉGORIELLES ===
# =====================================================
NUMERIC_FEATURES = [
    'Age', 'BMI', 'Appendix_Diameter', 'Body_Temperature', 'WBC_Count',
    'Neutrophil_Percentage', 'Hemoglobin', 'RDW', 'Thrombocyte_Count', 'CRP'
]

CATEGORICAL_FEATURES = [
    'Sex', 'Appendix_on_US', 'Migratory_Pain', 'Lower_Right_Abd_Pain',
    'Contralateral_Rebound_Tenderness', 'Coughing_Pain', 'Nausea',
    'Loss_of_Appetite', 'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine',
    'Dysuria', 'Psoas_Sign', 'Ipsilateral_Rebound_Tenderness', 'US_Performed',
    'Free_Fluids'
]

# =====================================================
# === FONCTION : OPTIMISATION DE LA MÉMOIRE DU DATAFRAME ===
# =====================================================
def optimize_memory(df):
    """
    Réduit la mémoire utilisée par le DataFrame en convertissant les colonnes numériques
    vers le type le plus petit possible et les colonnes 'object' à faible cardinalité en 'category'.
    """
    print(f"Mémoire avant optimisation : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    # === Optimisation des colonnes numériques ===
    for col in df.select_dtypes(include=[np.number]).columns:
        c_min, c_max = df[col].min(), df[col].max()
        if pd.api.types.is_integer_dtype(df[col]):
            # Conversion vers le plus petit type entier possible
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                info = np.iinfo(dtype)
                if info.min <= c_min and c_max <= info.max:
                    df[col] = df[col].astype(dtype)
                    break
        else:
            # Conversion vers float32 si possible
            if np.finfo(np.float32).min <= c_min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    # === Conversion des colonnes 'object' en 'category' si peu de modalités ===
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')

    after = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Mémoire après optimisation  : {after:.2f} MB")
    return df

# =====================================================
# === FONCTION : CHARGEMENT ET PRÉTRAITEMENT COMPLET DU DATASET ===
# =====================================================
def load_and_preprocess(filepath, target_col='Diagnosis', test_size=0.2, random_state=42):
    """
    Charge le fichier de données, optimise la mémoire, sépare les features et la cible,
    divise en train/test et crée un pipeline de prétraitement adapté.
    """
    # === Détection de l'extension pour choisir la méthode de chargement ===
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    print(f"Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # === Vérification de la présence de la colonne cible ===
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' introuvable. Colonnes : {df.columns.tolist()}")

    # === Séparation features/cible ===
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # === Vérification que la cible est bien numérique (0/1) ===
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("La cible doit être numérique (0/1).")

    # === Optimisation mémoire sur les features ===
    X = optimize_memory(X)

    # === Identification des colonnes numériques et catégorielles présentes dans X ===
    numeric_features = [col for col in NUMERIC_FEATURES if col in X.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in X.columns]

    # === Colonnes non listées → ajoutées comme catégorielles (sécurité) ===
    other_cols = [col for col in X.columns if col not in numeric_features and col not in categorical_features]
    if other_cols:
        print(f"Colonnes non listées (traitées comme catégorielles) : {other_cols}")
        categorical_features.extend(other_cols)

    print(f"  Numériques   ({len(numeric_features)}) : {numeric_features}")
    print(f"  Catégorielles({len(categorical_features)}) : {categorical_features}")

    # === Séparation en train/test (stratification sur la cible) ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Split → train : {len(X_train)} | test : {len(X_test)}")

    # === Pipeline pour les variables numériques : imputation + standardisation ===
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # === Pipeline pour les variables catégorielles : imputation + one-hot encoding ===
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # === Assemblage des pipelines dans un ColumnTransformer ===
    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_pipe, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # === Apprentissage du préprocesseur sur le train et transformation des deux jeux ===
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # === Sauvegarde du préprocesseur pour réutilisation future ===
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Préprocesseur sauvegardé → models/preprocessor.pkl")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

# =====================================================
# === FONCTION : RÉCUPÉRATION DES NOMS DE FEATURES APRÈS TRANSFORMATION ===
# =====================================================
def get_feature_names(preprocessor):
    """
    Retourne les noms des features après transformation par le préprocesseur.
    Utile pour interpréter les modèles downstream.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except:
        # Si la méthode n'existe pas, retourne des noms génériques
        n = preprocessor.transform(pd.DataFrame(columns=preprocessor.feature_names_in_)).shape[1]
        return [f"feature_{i}" for i in range(n)]

# =====================================================
# === FONCTION : SAUVEGARDE DES DONNÉES TRAITÉES ===
# =====================================================
def save_processed_data(X_train, X_test, y_train, y_test, feature_cols, output_path='data/processed/processed_data.joblib'):
    """
    Sauvegarde les données d'entraînement et de test traitées dans un fichier joblib.
    
    Paramètres:
    -----------
    X_train : array transformed
        Features d'entraînement transformées
    X_test : array transformed
        Features de test transformées
    y_train : Series
        Cible d'entraînement
    y_test : Series
        Cible de test
    feature_cols : list
        Noms des features après transformation
    output_path : str
        Chemin de sauvegarde du fichier joblib
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'n_samples_train': len(y_train),
        'n_samples_test': len(y_test)
    }
    
    joblib.dump(processed_data, output_path)
    print(f"✅ Données traitées sauvegardées → {output_path}")
    print(f"   Taille X_train : {X_train.shape}")
    print(f"   Taille X_test  : {X_test.shape}")
    print(f"   Features : {len(feature_cols)}")

# =====================================================
# === FONCTION : PIPELINE COMPLET - ORCHESTRATION ===
# =====================================================
def run_pipeline(raw_data_path='data/raw/dataset.xlsx', target_col='Diagnosis'):
    """
    Lance le pipeline complet de traitement des données :
    1. Chargement et prétraitement
    2. Récupération des noms de features
    3. Sauvegarde des données traitées et du préprocesseur
    
    Retour:
    -------
    dict : Dictionnaire avec les données traitées et métadonnées
    """
    print("\n" + "="*70)
    print(" PIPELINE DE TRAITEMENT DES DONNÉES - APPENDICITE PÉDIATRIQUE")
    print("="*70 + "\n")
    
    # Étape 1 : Chargement et prétraitement
    print("[1/3] Chargement et prétraitement des données...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(
        raw_data_path, 
        target_col=target_col
    )
    
    # Étape 2 : Récupération des noms de features
    print("\n[2/3] Récupération des noms de features...")
    feature_cols = get_feature_names(preprocessor)
    print(f"✅ {len(feature_cols)} features identifiées")
    
    # Étape 3 : Sauvegarde des données traitées
    print("\n[3/3] Sauvegarde des données traitées...")
    save_processed_data(
        X_train, X_test, y_train, y_test, feature_cols,
        output_path='data/processed/processed_data.joblib'
    )
    
    print("\n" + "="*70)
    print(" ✅ PIPELINE COMPLÉTÉ AVEC SUCCÈS")
    print("="*70 + "\n")
    
    # Retour des métadonnées
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'n_samples_train': len(y_train),
        'n_samples_test': len(y_test)
    }

# =====================================================
# === EXEMPLE D'UTILISATION DU MODULE EN LANCEMENT DIRECT ===
# =====================================================
if __name__ == '__main__':
    # Lance le pipeline complet et sauvegarde les résultats
    results = run_pipeline(
        raw_data_path='data/raw/dataset.xlsx',
        target_col='Diagnosis'
    )
    
    print("\n📊 RÉSUMÉ DU PIPELINE :")
    print(f"   • Données d'entraînement : {results['X_train'].shape}")
    print(f"   • Données de test : {results['X_test'].shape}")
    print(f"   • Features : {results['n_features']}")
    print(f"   • Distribution train : {len(results['y_train'])} échantillons")
    print(f"   • Distribution test : {len(results['y_test'])} échantillons")