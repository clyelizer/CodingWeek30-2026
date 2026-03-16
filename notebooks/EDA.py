#importer les packages
import numpy as np 
import pandas as pd
from scipy import stats

#ouvrir la base de données
df=pd.read_excel("/content/app_data.xlsx")
df
df.shape
df.info
df.describe()

#supprimer les lignes repetées
df.duplicated().sum()

#découverte des valeurs manquantes
df.isnull().sum()

#gestion des valeurs manquantes
  
  #gestion des variables manquantes de type faible et modéré
  
      #gestion des variables de type numérique
df['Appendix_Diameter'].fillna(df['Appendix_Diameter'].median(),inplace=True)
df['Neutrophil_Percentage'].fillna(df['Neutrophil_Percentage'].median(),inplace=True)
df['Alvarado_Score'].fillna(df['Alvarado_Score'].median(),inplace=True)
df['Paedriatic_Appendicitis_Score'].fillna(df['Paedriatic_Appendicitis_Score'].median(),inplace=True)
df['BMI'].fillna(df['BMI'].median(),inplace=True)
df['Height'].fillna(df['Height'].median(),inplace=True)
df['RDW'].fillna(df['RDW'].median(),inplace=True)
df['US_Number'].fillna(df['US_Number'].median(),inplace=True)
df['Hemoglobin'].fillna(df['Hemoglobin'].median(),inplace=True)
df['Thrombocyte_Count'].fillna(df['Thrombocyte_Count'].median(),inplace=True)
df['RBC_Count'].fillna(df['RBC_Count'].median(),inplace=True)
df['CRP'].fillna(df['CRP'].median(),inplace=True)
df['Body_Temperature'].fillna(df['Body_Temperature'].median(),inplace=True)
df['WBC_Count'].fillna(df['WBC_Count'].median(),inplace=True)
df['Length_of_Stay'].fillna(df['Length_of_Stay'].median(),inplace=True)
df['Weight'].fillna(df['Weight'].median(),inplace=True)
df['Age'].fillna(df['Age'].median(),inplace=True)

     #gestion des variables catégorielles
colonnes_cat = ['RBC_in_Urine','Ketones_in_Urine','WBC_in_Urine','Ipsilateral_Rebound_Tenderness'
    'Sex', 'Diagnosis', 'Severity', 'Management', 'Neutrophilia',
    'Migratory_Pain', 'Lower_Right_Abd_Pain', 'Contralateral_Rebound_Tenderness',
    'Coughing_Pain', 'Nausea', 'Loss_of_Appetite', 'Dysuria',
    'Stool', 'Peritonitis', 'Psoas_Sign', 'Appendix_on_US',
    'US_Performed', 'Free_Fluids','Diagnosis_Presumptive']

for col in colonnes_cat:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        
  #gestion des variables de type critique
colonnes_critiques = [
    'Abscess_Location', 'Gynecological_Findings', 'Conglomerate_of_Bowel_Loops',
    'Segmented_Neutrophils', 'Ileus', 'Perfusion', 'Enteritis', 'Appendicolith',
    'Coprostasis', 'Perforation', 'Appendicular_Abscess', 'Bowel_Wall_Thickening',
    'Lymph_Nodes_Location', 'Target_Sign', 'Meteorism', 'Pathological_Lymph_Nodes',
    'Appendix_Wall_Layers', 'Surrounding_Tissue_Reaction']

# Convertir en binaire : 1 si valeur présente, 0 si manquant
for col in colonnes_critiques:
    df[col] = df[col].notna().astype(int)
    
#valeurs aberrantes
  #Détection par IQR
colonnes_numeriques = ['Age', 'BMI', 'CRP', 'WBC_Count', 'Hemoglobin', 
                        'Body_Temperature', 'RDW', 'Appendix_Diameter',
                        'Alvarado_Score', 'Paedriatic_Appendicitis_Score',
                        'Length_of_Stay', 'Weight', 'Height', 'RBC_Count',
                        'Thrombocyte_Count', 'Neutrophil_Percentage']

for col in colonnes_numeriques:
    s = df[col].dropna()
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    bb, bh = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n = ((s < bb) | (s > bh)).sum()
    print(f'{col}: {n} outliers | bornes=[{bb:.2f}, {bh:.2f}]')

  #Détection par Z-scores
for col in colonnes_numeriques:
    s = df[col].dropna()
    z = np.abs(stats.zscore(s))
    print(f'{col}: {(z > 3).sum()} outliers (z>3)')

  #suppression des valeurs biologiques impossibles
df = df[df['Body_Temperature'] > 34]
df = df[df['Hemoglobin'] <= 20]
df = df[df['RDW'] <= 30]
df = df[df['RBC_Count'] <= 8]

   #Winnorisation
colonnes_winsor = ['BMI', 'WBC_Count', 'Length_of_Stay', 
                   'Thrombocyte_Count', 'Appendix_Diameter']

for col in colonnes_winsor:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    borne_basse = Q1 - 1.5 * IQR
    borne_haute = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=borne_basse, upper=borne_haute)

   #Log-transformation de CRP
df['CRP_log'] = np.log1p(df['CRP'])

