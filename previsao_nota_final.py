"""
P3 - Machine Learning: Previsão da Nota Final de Alunos
=========================================================

Objetivo:
    Criar uma base de dados fake de alunos e um modelo de Machine Learning
    capaz de prever a nota final com base em:
        - horas de estudo semanais
        - frequência (%)
        - atividades entregues (%)
        - nota da avaliação anterior

    Ao final, calculamos média, desvio padrão e intervalo de erro das
    previsões, e avaliamos se o desempenho do modelo é aceitável.

Integrante:
    - João Pedro Machado
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# 1. GERAÇÃO DA BASE DE DADOS FAKE
# ---------------------------------------------------------------------------
np.random.seed(42)
N_ALUNOS = 500

horas_estudo = np.round(np.random.normal(loc=8, scale=3, size=N_ALUNOS).clip(0, 20), 1)
frequencia = np.round(np.random.normal(loc=80, scale=12, size=N_ALUNOS).clip(30, 100), 1)
atividades_entregues = np.round(np.random.normal(loc=75, scale=15, size=N_ALUNOS).clip(0, 100), 1)
nota_anterior = np.round(np.random.normal(loc=6.5, scale=1.5, size=N_ALUNOS).clip(0, 10), 1)

# Nota final = combinação linear das variáveis + ruído aleatório (simula
# a variabilidade real de desempenho de um aluno)
ruido = np.random.normal(loc=0, scale=0.6, size=N_ALUNOS)

nota_final = (
    0.18 * horas_estudo
    + 0.035 * frequencia
    + 0.025 * atividades_entregues
    + 0.35 * nota_anterior
    + ruido
)
nota_final = np.round(np.clip(nota_final, 0, 10), 1)

df = pd.DataFrame({
    "aluno_id": range(1, N_ALUNOS + 1),
    "horas_estudo": horas_estudo,
    "frequencia": frequencia,
    "atividades_entregues": atividades_entregues,
    "nota_anterior": nota_anterior,
    "nota_final": nota_final,
})

df.to_csv("base_alunos.csv", index=False)
print("Base de dados gerada: base_alunos.csv")
print(df.head(10).to_string(index=False))
print(f"\nTotal de alunos: {len(df)}")

# ---------------------------------------------------------------------------
# 2. SEPARAÇÃO TREINO / TESTE
# ---------------------------------------------------------------------------
features = ["horas_estudo", "frequencia", "atividades_entregues", "nota_anterior"]
X = df[features]
y = df["nota_final"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 3. TREINAMENTO DO MODELO (Regressão Linear)
# ---------------------------------------------------------------------------
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("\nCoeficientes do modelo (Regressão Linear):")
for f, c in zip(features, modelo.coef_):
    print(f"  {f}: {c:.4f}")
print(f"  Intercepto: {modelo.intercept_:.4f}")

# ---------------------------------------------------------------------------
# 4. ESTATÍSTICAS DOS ERROS (RESÍDUOS) DE PREVISÃO
# ---------------------------------------------------------------------------
erros = y_test.values - y_pred  # erro = valor real - valor previsto

media_erro = np.mean(erros)
desvio_padrao_erro = np.std(erros, ddof=1)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Intervalo de erro (95% de confiança), assumindo distribuição ~normal dos resíduos
intervalo_95_inferior = media_erro - 1.96 * desvio_padrao_erro
intervalo_95_superior = media_erro + 1.96 * desvio_padrao_erro

print("\n" + "=" * 60)
print("ESTATÍSTICAS DO MODELO")
print("=" * 60)
print(f"Média do erro (viés):          {media_erro:.4f}")
print(f"Desvio padrão do erro:         {desvio_padrao_erro:.4f}")
print(f"Erro Médio Absoluto (MAE):     {mae:.4f}")
print(f"Raiz do Erro Quadrático (RMSE):{rmse:.4f}")
print(f"R² (coeficiente de determinação): {r2:.4f}")
print(f"Intervalo de erro (95%): [{intervalo_95_inferior:.4f}, {intervalo_95_superior:.4f}]")

# Salva estatísticas em arquivo texto
with open("estatisticas.txt", "w", encoding="utf-8") as f:
    f.write("ESTATÍSTICAS DO MODELO DE PREVISÃO DE NOTA FINAL\n")
    f.write("=" * 55 + "\n")
    f.write(f"Média do erro (viés):              {media_erro:.4f}\n")
    f.write(f"Desvio padrão do erro:             {desvio_padrao_erro:.4f}\n")
    f.write(f"Erro Médio Absoluto (MAE):         {mae:.4f}\n")
    f.write(f"Raiz do Erro Quadrático (RMSE):    {rmse:.4f}\n")
    f.write(f"R² (coeficiente de determinação):  {r2:.4f}\n")
    f.write(f"Intervalo de erro (95%%):           [{intervalo_95_inferior:.4f}, {intervalo_95_superior:.4f}]\n")

# ---------------------------------------------------------------------------
# 5. GRÁFICOS
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")

# 5.1 Real vs Previsto
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k", color="#4C72B0")
lims = [0, 10]
plt.plot(lims, lims, "r--", label="Previsão perfeita")
plt.xlabel("Nota Real")
plt.ylabel("Nota Prevista")
plt.title("Nota Real vs. Nota Prevista")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_real_vs_previsto.png", dpi=150)
plt.close()

# 5.2 Distribuição dos erros (resíduos)
plt.figure(figsize=(7, 6))
sns.histplot(erros, kde=True, color="#DD8452", bins=15)
plt.axvline(media_erro, color="red", linestyle="--", label=f"Média = {media_erro:.2f}")
plt.xlabel("Erro (Real - Previsto)")
plt.ylabel("Frequência")
plt.title("Distribuição dos Erros de Previsão")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_distribuicao_erros.png", dpi=150)
plt.close()

# 5.3 Correlação entre variáveis
plt.figure(figsize=(7, 6))
corr = df[features + ["nota_final"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlação entre Variáveis")
plt.tight_layout()
plt.savefig("grafico_correlacao.png", dpi=150)
plt.close()

# 5.4 Importância das variáveis (Random Forest, para comparação/robustez)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

importancias = pd.Series(rf.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7, 5))
importancias.plot(kind="barh", color="#55A868")
plt.xlabel("Importância")
plt.title("Importância das Variáveis (Random Forest)")
plt.tight_layout()
plt.savefig("grafico_importancia_variaveis.png", dpi=150)
plt.close()

print("\nComparação com Random Forest (modelo auxiliar):")
print(f"  MAE (Random Forest): {mae_rf:.4f}")
print(f"  R²  (Random Forest): {r2_rf:.4f}")

print("\nGráficos salvos:")
print("  - grafico_real_vs_previsto.png")
print("  - grafico_distribuicao_erros.png")
print("  - grafico_correlacao.png")
print("  - grafico_importancia_variaveis.png")
print("\nProcesso concluído com sucesso.")
