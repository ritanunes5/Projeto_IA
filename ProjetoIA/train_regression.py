'''
train_regression.py

Módulo 2 
Modelo usado: Linear Regression
'''

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)


# carregamento de dados
df = pd.read_csv('processed_lisboa_porto_air_quality.csv', sep=';')

# variáveis usadas para prever a concentração de NO2
features = ['CO', 'O3', 'PM10', 'PM2.5', 'SO2',
            'temperature_c', 'humidity_percent', 'wind_speed_kmh',
            'precipitation_mm', 'C6H6', 'NOx']

# variável que queremos prever: NO2
target = 'NO2'

# Remove linhas onde o target é NaN
df_reg = df[features + [target]].dropna(subset=[target])

# definição de x e y
X = df_reg[features]
y = df_reg[target]

print(f'Dataset (após remover NaN no target): {X.shape[0]} registos, {X.shape[1]} features')
print(f'NO2 — média: {y.mean():.2f} | std: {y.std():.2f} | min: {y.min():.2f} | max: {y.max():.2f}\n')


# Pré processamento


# Imputação dos missing values nas features com a mediana
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

# Normalização dos valores
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# divisão entre dataset de treino/teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    # dois arrays(x_scaled e y)
    X_scaled, y, test_size=0.2, random_state=42
)

print(f'Treino: {X_train.shape[0]} registos')
print(f'Teste : {X_test.shape[0]} registos\n')

# treino do modelo

modelos = {
    'Linear Regression':      LinearRegression()
}

resultados = []

for nome, modelo in modelos.items():

    # treino do modelo com os dados de treino a partir dos dados x de treino e y de treino
    modelo.fit(X_train, y_train)

    # previsão do resultado a partir do x de teste
    y_pred = modelo.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred) # Erro médio absoluto
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Raiz do erro quadrático médio
    r2   = r2_score(y_test, y_pred) #   Coeficiente de Determinação proporção da variância explicada pelo modelo (0-1)


    resultados.append({
        'modelo': nome,
        'MAE':  round(mae,  4),
        'RMSE': round(rmse, 4),
        'R2':   round(r2,   4),
    })

    print(f' {nome}')
    print(f'  MAE  : {mae:.4f} µg/m³')
    print(f'  RMSE : {rmse:.4f} µg/m³')
    print(f'  R²   : {r2:.4f}\n')

    # Gráfico real/previsto
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Previsão perfeita')
    ax.set_xlabel('NO2 Real (µg/m³)')
    ax.set_ylabel('NO2 Previsto (µg/m³)')
    ax.set_title(f'Real / Previsto — {nome}', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fname = f'regression_scatter_{nome.replace(" ", "_").lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()

    # Guarda o modelo
    with open(f'modelo_reg_{nome.replace(" ", "_").lower()}.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    print(f' Modelo guardado em modelo_reg_{nome.replace(" ", "_").lower()}.pkl\n')


# Guarda scaler e imputer da regressão
with open('scaler_reg.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('imputer_reg.pkl', 'wb') as f:
    pickle.dump(imputer, f)


df_metrics_reg = pd.DataFrame(resultados)
df_metrics_reg['tarefa'] = 'regressao'

# Adicionar valores obtidos ao métrics.csv gerado pelo train_classification.py
try:
    df_metrics_clf = pd.read_csv('metrics.csv',sep=";")
    df_all = pd.concat([df_metrics_clf, df_metrics_reg], ignore_index=False)
except FileNotFoundError:
    df_all = df_metrics_reg
df_all.to_csv('metrics.csv', index=False,sep=";")
print('\n Métricas acrescentadas em metrics.csv')



