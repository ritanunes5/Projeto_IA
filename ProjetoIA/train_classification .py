'''
train_classification.py

Módulo 2 
Modelos usados: Logistic Regression e Random Forest
'''

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

np.random.seed(42)


# carregamento de dados
df = pd.read_csv('processed_lisboa_porto_air_quality.csv', sep=';')

# variáveis que vamos usar para determinar se a qualidade do ar é boa ou não
features = ['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2',
            'temperature_c', 'humidity_percent', 'wind_speed_kmh', 'precipitation_mm']

#variável que procuramos determinar (se a qualidade do ar é boa ou não)
target = 'air_quality_good'

# definição de x e y
X = df[features]
y = df[target]


print(f'Dataset: {X.shape[0]} registos, {X.shape[1]} features')
print(f'Distribuição da label:\n{y.value_counts()}\n')


# Pré-processamento


# Imputação dos missing values com a mediana
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

# Normalização dos valores 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# divisão entre dataset de treino/teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    # dois arrays(x_scaled e y)
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#
print(f'Treino: {X_train.shape[0]} registos')
print(f'Teste : {X_test.shape[0]} registos\n')


# Treino dos modelos de classificação

# dicionário dos modelos que vamos usar
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
}

# resultados
resultados = []

# para ambos os modelos
for nome, modelo in modelos.items():

    # treino do modelo com os dados de treino
    modelo.fit(X_train, y_train)

    # previsão do resultado
    y_pred = modelo.predict(X_test)

    # calculo de métricas básicas
    acc  = accuracy_score(y_test, y_pred) # exatidão (%)
    prec = precision_score(y_test, y_pred) # precisão (tp/(tp+fp)) 
    rec  = recall_score(y_test, y_pred) # sensibilidade (tp/(tp+fn))
    f1   = f1_score(y_test, y_pred) # F1-score (2*prediction*recall/ prediction+recall) média harmónica da precision e recall


    # obtenção de modelo e métricas arredondadas
    resultados.append({
        'modelo': nome,
        'accuracy':  round(acc,  4),
        'precision': round(prec, 4),
        'recall':    round(rec,  4),
        'f1_score':  round(f1,   4),
    })

    print(f'{nome}')
    print(f'  Accuracy : {acc:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall   : {rec:.4f}')
    print(f'  F1-Score : {f1:.4f}\n')

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Má', 'Boa'])
    disp.plot(colorbar=False)
    plt.title(f'Matriz de Confusão — {nome}', fontweight='bold')
    plt.savefig(f'confusion_matrix_{nome.replace(" ", "_").lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Guarda o modelo
    with open(f'modelo_{nome.replace(" ", "_").lower()}.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    print(f'  Modelo guardado em modelo_{nome.replace(" ", "_").lower()}.pkl\n')

# Guarda também o scaler e imputer, necessários para usar os modelos 
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)


# Comparação e escolha do modelo final

# criação de um dataframe com os resultados (modelo, accuracy, precision, recall, f1_score)
df_metrics = pd.DataFrame(resultados)
print('── Comparação de Modelos ──')
print(df_metrics.to_string(index=False))

# Guardar métricas
df_metrics['tarefa'] = 'classificacao'
df_metrics.to_csv('metrics.csv', index=False, sep=";")
print('\n Métricas guardadas em metrics.csv')

# Justificação automática com base no F1
melhor = df_metrics.loc[df_metrics['f1_score'].idxmax(), 'modelo']
print(f'\n Modelo final escolhido: {melhor}')
print('  Justificação: escolhido com base no F1-Score mais alto,')
print('  métrica mais equilibrada em datasets com possível desbalanceamento de classes.')

# Gráfico de comparação entre os modelos
fig, ax = plt.subplots(figsize=(8, 5))
metricas_plot = ['accuracy', 'precision', 'recall', 'f1_score']
x = np.arange(len(metricas_plot))
width = 0.35

for i, (_, row) in enumerate(df_metrics.iterrows()):
    ax.bar(x + i * width, [row[m] for m in metricas_plot], width, label=row['modelo'], alpha=0.85)

ax.set_xticks(x + width / 2)
ax.set_xticklabels(metricas_plot)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Comparação de Métricas — Classificação', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('comparacao_classificacao.png', dpi=150, bbox_inches='tight')
plt.show()

print(' Gráfico comparativo guardado em comparacao_classificacao.png')
