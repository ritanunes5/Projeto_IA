'''
rules_engine.py 

Módulo 1 
'''
# imports

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import timedelta
from itertools import product

# Sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance


warnings.filterwarnings('ignore')
np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

print('Bibliotecas carregadas')

# Carrega os dados
df = pd.read_csv('processed_lisboa_porto_air_quality.csv', sep=";")
print(df.head())

#motor de inferência
def calcular_news2(wind_speed_kmh , precipitation_mm, temperature_c, humidity_percent, pm10, O3, no2_concentration ):

    # lista dos avisos e recomendações
    avisos = []
    recomendacoes = []

    # regra 1
    if wind_speed_kmh > 70:
        avisos.append("Rajada de vento")
        recomendacoes.append("Não sair de casa")
    
    # regra 2
    if precipitation_mm > 0:
        avisos.append("Aviso de precipitação")
        recomendacoes.append("Levar guarda chuva")

    # regra 3
    if temperature_c > 30 and humidity_percent < 30:
        avisos.append("Aviso de incêndio")
        recomendacoes.append("Não fazer queimadas")

    # regra 4
    if temperature_c < 0 and precipitation_mm >0:
        avisos.append("Aviso de neve")
        recomendacoes.append("Vestir varias camadas de roupa")

    # regra 5
    if temperature_c < 10 and wind_speed_kmh > 5:
        avisos.append("Possibilidade de congelamento")
        recomendacoes.append("Manter-se seco")

    # regra 6
    if humidity_percent > 95 and wind_speed_kmh < 5:
        avisos.append("Nevoeiro denso")
        recomendacoes.append("Conduzir com cuidado")

    # regra 7
    if pm10 > 50:
        avisos.append("Alerta de partículad inaláveis")
        recomendacoes.append("Usar máscara")

    # regra 8
    if O3 > 180 and temperature_c > 30 :
        avisos.append("Concentração de Ozono Elevada")
        recomendacoes.append("Evitar exercício ao ar livre")

    # regra 9
    if no2_concentration > 200 :
        avisos.append("Concentração de NO2 Elevada")
        recomendacoes.append("Mantenha as janelas fechadas")

    # regra 10
    if precipitation_mm == 0 and humidity_percent < 20 and temperature_c > 35 :
        avisos.append("Aviso de seca extrema")
        recomendacoes.append("Evitar o desperdício de água")

    return avisos, recomendacoes

# aplicação de regras para todo o dataset    
resultado_new2 = df.apply(
    lambda r:  calcular_news2(r["wind_speed_kmh"], r["precipitation_mm"], r["temperature_c"], r["humidity_percent"], r["PM10"],r["O3"],r["NO2"] ),axis=1
)

#criação de cópia do dataset
df_alterado = df.copy()

#criação de duas novas colunas: avisos e recomendações
df_alterado["avisos"] = resultado_new2.apply(lambda x: x[0])
df_alterado["recomendações"] = resultado_new2.apply(lambda x: x[1])

print(df_alterado.head())

# criação de uma nova tabela com os avisos e recomendações para cada tuplo do dataset
df_alterado.to_csv("air_quality_plus_avisos_e_recomendacoes.csv",index=False, sep=";", encoding="utf-8-sig")










