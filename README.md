# P3 — Previsão da Nota Final com Machine Learning

Atividade que cria uma base de dados fake de alunos e um modelo de Machine
Learning para prever a nota final com base em horas de estudo, frequência,
atividades entregues e nota anterior.

## Integrante
- João Pedro Machado

## Conteúdo do repositório
- `previsao_nota_final.py` — script completo: gera a base, treina o modelo
  (Regressão Linear + Random Forest de comparação), calcula estatísticas e
  gera os gráficos.
- `base_alunos.csv` — base de dados fake gerada (500 alunos).
- `estatisticas.txt` — estatísticas dos erros de previsão.
- `grafico_real_vs_previsto.png`, `grafico_distribuicao_erros.png`,
  `grafico_correlacao.png`, `grafico_importancia_variaveis.png` — gráficos.
- `P3_Previsao_Nota_Final.pdf` — relatório final (código, gráficos,
  estatísticas e análise).

## Como rodar

O projeto deve ser executado no **compilador online OnlineGDB**, utilizando a linguagem **Python 3**.

1. Acesse o [OnlineGDB](https://www.onlinegdb.com/online_python_compiler).
2. Selecione a linguagem **Python 3**.
3. Copie o código do arquivo `previsao_nota_final.py` e cole no editor.
4. Clique em **Run** para executar o programa.

- **Linguagem:** Python 3
- **Compilador:** OnlineGDB

## Resumo dos resultados
| Métrica | Valor |
|---|---|
| Média do erro (viés) | -0,0458 |
| Desvio padrão do erro | 0,5476 |
| MAE | 0,4518 |
| RMSE | 0,5467 |
| R² | 0,6658 |
| Intervalo de erro (95%) | [-1,12 ; 1,03] |

O modelo explica ~67% da variação da nota final e erra, em média, menos de
meio ponto (escala 0–10). Desempenho considerado aceitável para o problema
proposto.
