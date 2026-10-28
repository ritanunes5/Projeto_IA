'''
bayes_alerts.py

Modulo 1
'''

import pandas as pd
import numpy as np
from itertools import product

# Discretização 

def discretizar(df: pd.DataFrame) -> pd.DataFrame:
    
    #Converte as variáveis contínuas em classes
    
    d = pd.DataFrame(index=df.index)

    # Vento: fraco / moderado / forte  (moderado = 20 / 70 km/h)
    d["wind_speed"] = pd.cut(
        df["wind_speed_kmh"],
        bins=[-np.inf, 20, 70, np.inf],
        labels=["fraco", "moderado", "forte"]
    )

    # Precipitação: sem / com  (sem chuva = 0 mm)
    d["precipitation"] = pd.cut(
        df["precipitation_mm"],
        bins=[-np.inf, 0, np.inf],
        labels=["sem", "com"]
    )

    # Nó alvo: tempestade
    # se existe precipitação e vento moderado/forte 
    d["tempestade"] = np.where(
        (df["wind_speed_kmh"] > 20) & (df["precipitation_mm"] > 0),
        "sim", "nao"
    )

    return d.dropna()



# classe da rede bayesiana

class BayesianNetwork:

    #iniciar os nós
    def __init__(self):
        self.cpds = {}
        self.parents = {
            "wind_speed":    [], # nó raiz
            "precipitation": [], # nó raiz
            "tempestade":    ["wind_speed", "precipitation"] # dependente dos dois nós anteriores
        }
        self.categories = {}

    # Aprendizagem de  probabilidades
    def fit(self, df: pd.DataFrame, alpha: float = 1.0):

        
        # conversão do dataset para string
        df = df.astype(str)

        # para cada nó,  guardar as categorias possíveis
        for node in self.parents:
            self.categories[node] = sorted(df[node].unique().tolist())
        
        # estrutura dos CPDs
        for node, pais in self.parents.items():
            self.cpds[node] = {}
            vals_node = self.categories[node]

        # Estima as CPDs por Maximum Likelihood com suavização de Laplace (alpha).

            # Nó raiz:  P(node)
            if not pais:
                # nº de ocorrências
                counts = df[node].value_counts()

                #probabilidade com suavisação de laplace
                total  = counts.sum() + alpha * len(vals_node)
                self.cpds[node][()] = {
                    v: (counts.get(v, 0) + alpha) / total for v in vals_node
                }

            # Nó com pais:  P(node | pais)
            else:
                
                #valores dos nós com pais
                vals_pais = [self.categories[p] for p in pais]

                # todas as combinações possíveis dos nós pais
                for combo_pais in product(*vals_pais):

                    # filtragem de dados, apenas que correspondam á combinação
                    mask = pd.Series([True] * len(df), index=df.index)
                    for p, v in zip(pais, combo_pais):
                        mask &= (df[p] == v)

                    # obtenção do valor do nó
                    subset = df[mask][node]

                    # contagem de ocorrência
                    counts  = subset.value_counts()

                    # probabilidade com suavização
                    total   = counts.sum() + alpha * len(vals_node)
                    self.cpds[node][combo_pais] = {
                        v: (counts.get(v, 0) + alpha) / total for v in vals_node
                    }

        print("CPDs estimadas com sucesso")
        return self
    
    # inferência Bayesiana
    def query_tempestade(self, evidence: dict) -> dict:
        
        # Inferência por enumeração:

        #P(tempestade | wind_speed, precipitation observados).
        ev    = {k: str(v) for k, v in evidence.items()}
        probs = {}

        # iteneração sobre todos os estados possiveis da tempetade (sim, não)
        for valor in self.categories["tempestade"]:

            #iniciação do variável probabilidade
            prob = 0.0

            # nós pais 
            nos_raiz = ["wind_speed", "precipitation"]
            
            # construção dos valores a itenerar
            vals_iterar = []

            # para cada nó pai
            for n in nos_raiz:
                 # caso a variável já esteja em evidência
                if n in ev:
                    vals_iterar.append([ev[n]])
                # caso seja uma variável desconhecida
                else:
                    vals_iterar.append(self.categories[n])

            # enumeração (geração de todas as combinações possíveis)
            for combo in product(*vals_iterar):

                # P(wind_speed) e P(precipitation) — só usados quando marginalizamos
                # probabilidade do vento
                p_wind  = self.cpds["wind_speed"][()].get(combo[0],  1e-9) if "wind_speed"    not in ev else 1.0

                # probabilidade da chiva
                p_prec  = self.cpds["precipitation"][()].get(combo[1], 1e-9) if "precipitation" not in ev else 1.0

                # P(tempestade | wind_speed, precipitation)
                # Distribuição de Prpbabilidade condicional da tempestade
                key    = (combo[0], combo[1])
                p_temp = self.cpds["tempestade"].get(key, {}).get(valor, 1e-9)

                # combinação
                prob += p_wind * p_prec * p_temp

            # guardar resultados por classe
            probs[valor] = prob

        # Normalizar (P(sim) +  P(não) = 1)
        total = sum(probs.values())
        if total > 0:
            probs = {k: round(v / total, 4) for k, v in probs.items()}
        return probs



# main 
if __name__ == "__main__":

    # Carregar os dados e discretizar
    df_raw  = pd.read_csv("processed_lisboa_porto_air_quality.csv", sep=";")
    df_disc = discretizar(df_raw)

    print("=" * 60)
    print("DISTRIBUIÇÃO DAS VARIÁVEIS DISCRETIZADAS")
    print("=" * 60)
    for col in df_disc.columns:
        print(f"\n{col}:\n{df_disc[col].value_counts().to_string()}")

    # Treina a rede
    print("\n" + "=" * 60)
    print("ESTIMAÇÃO DAS CPDs")
    print("=" * 60)
    bn = BayesianNetwork()
    bn.fit(df_disc)

    # Aplicar ao dataset completo
    print("\nINFERÊNCIA SOBRE O DATASET COMPLETO")
    
    resultados = []
    for _, row in df_disc.iterrows():
        ev  = {"wind_speed": str(row["wind_speed"]), "precipitation": str(row["precipitation"])}
        res = bn.query_tempestade(ev)
        mais_provavel = max(res, key=res.get)
        resultados.append({
            "P_tempestade_sim": res.get("sim", 0),
            "P_tempestade_nao": res.get("nao", 0),
            "Tempestade_Prevista": mais_provavel
        })

    df_res   = pd.DataFrame(resultados)
    df_final = pd.concat([df_raw, df_disc, df_res], axis=1)

    total  = len(df_final)
    tempes = (df_final["Tempestade_Prevista"] == "sim").sum()
    print(f"\nTotal de registos      : {total}")
    print(f"Com Tempestade         : {tempes}  ({100*tempes/total:.1f} %)")
    print(f"Sem Tempestade         : {total-tempes}  ({100*(total-tempes)/total:.1f} %)")

    # primeiros 10 registros
    cols_show = ["wind_speed_kmh", "precipitation_mm", "wind_speed", "precipitation",
                 "tempestade", "P_tempestade_sim", "Tempestade_Prevista"]
    print("\nPrimeiros 10 registos:")
    print(df_final[cols_show].head(10).to_string(index=False))

    # criação de um csv com os resultados 
    df_final.to_csv("bayes_resultados.csv", index=False, sep=";", encoding="utf-8-sig")
    print("\n Resultados guardados em bayes_resultados.csv")
