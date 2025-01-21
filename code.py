import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as opt
import matplotlib.pyplot as plt
import mplcyberpunk

# Aplica o estilo Cyberpunk
plt.style.use("cyberpunk")

# 🔹 Função para validar tickers
def validar_tickers(tickers):
    dados_teste = yf.download(tickers, period="1d")
    if dados_teste.empty:
        return False
    return True

# 🔹 Pergunta ao usuário quantos ativos serão incluídos
while True:
    try:
        num_ativos = int(input("Quantos ativos deseja incluir no portfólio? "))
        if num_ativos <= 0:
            print("Por favor, insira um número positivo.")
            continue
        break
    except ValueError:
        print("Entrada inválida. Por favor, insira um número inteiro.")

# 🔹 Recebe os tickers do usuário
ativos = []
for i in range(num_ativos):
    while True:
        ticker = input(f"Digite o ticker do ativo {i+1}: ").upper().strip()
        if ticker in ativos:
            print("Ticker já inserido. Por favor, insira um ticker diferente.")
            continue
        if validar_tickers(ticker):
            ativos.append(ticker)
            break
        else:
            print("Ticker inválido ou dados indisponíveis. Tente novamente.")

# 🔹 Pergunta ao usuário o período de análise
while True:
    start_date = input("Digite a data de início no formato AAAA-MM-DD (ex: 2020-01-01): ")
    end_date = input("Digite a data de término no formato AAAA-MM-DD (ex: 2023-12-31): ")
    try:
        dados = yf.download(ativos, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
        if dados.empty:
            print("Dados não encontrados para o período e ativos selecionados. Tente novamente.")
            continue
        break
    except Exception as e:
        print(f"Erro ao baixar os dados: {e}. Tente novamente.")

# 🔹 Calcular os retornos diários
retornos_diarios = dados.pct_change().dropna()

# 🔹 Calcular retornos médios anuais e matriz de covariância
retornos_anuais = retornos_diarios.mean() * 252  # Convertendo para base anual
cov_matrix = retornos_diarios.cov() * 252       # Matriz de covariância anualizada

# 🔹 Função objetivo: Minimizar o risco (variância do portfólio)
def risco_portfolio(w, cov_matrix):
    return w.T @ cov_matrix @ w

# 🔹 Restrições do problema
def retorno_minimo(w, retorno_esperado):
    return np.dot(retornos_anuais, w) - retorno_esperado

def soma_pesos(w):
    return np.sum(w) - 1

# 🔹 Definir limites para os pesos (nenhum peso pode ser negativo)
bounds = [(0, 1) for _ in range(len(ativos))]

# 🔹 Chute inicial para os pesos (distribuição igualitária)
w_init = np.ones(len(ativos)) / len(ativos)

# 🔹 Pergunta ao usuário o retorno mínimo desejado
while True:
    try:
        retorno_desejado = float(input("Digite o retorno mínimo anual desejado (em decimal, ex: 0.15 para 15%): "))
        if retorno_desejado <= 0:
            print("O retorno desejado deve ser positivo.")
            continue
        break
    except ValueError:
        print("Entrada inválida. Por favor, insira um número decimal.")

# 🔹 Montar as restrições para o solver
constraints = [
    {'type': 'eq', 'fun': soma_pesos},
    {'type': 'ineq', 'fun': lambda w: retorno_minimo(w, retorno_desejado)}
]

# 🔹 Resolver o problema de otimização usando SciPy
result = opt.minimize(
    risco_portfolio,
    w_init,
    args=(cov_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# 🔹 Verificar se a otimização foi bem-sucedida
if not result.success:
    print("A otimização não convergiu. Tente ajustar as restrições ou os ativos selecionados.")
    print("Mensagem de erro:", result.message)
    exit()

# 🔹 Obter os valores ótimos de alocação
pesos_otimos = result.x

# 🔹 Criar DataFrame para exibição dos resultados
df_result = pd.DataFrame({
    "Ativo": ativos,
    "Alocação Ótima (%)": pesos_otimos * 100  # Convertendo para percentual
})

# 🔹 Exibir a alocação ótima do portfólio
print("\n### Alocação Ótima do Portfólio ###")
print(df_result)

# 🔹 Plotar a distribuição ótima do portfólio (pie chart)
plt.figure(figsize=(10, 6))
patches, texts, autotexts = plt.pie(
    pesos_otimos,
    labels=None,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=1.1
)

plt.title("Distribuição Ótima do Portfólio (Teoria de Markowitz)", fontweight="bold", fontsize=14)

# Adiciona a legenda separada
plt.legend(patches, ativos, title="Ativos", loc="upper right", bbox_to_anchor=(1.3, 1))

# Aplica efeito Glow do Cyberpunk
mplcyberpunk.add_glow_effects()

plt.tight_layout()
plt.show()
