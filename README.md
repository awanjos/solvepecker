# solvepecker
Tactical Forwarder Planning in Forestry Supply Chain

# IMPORTAÇÃO DE BIBLIOTECAS

import numpy as np
import pandas as pd
import scipy as sp
import cvxpy as cp
import datetime

from scipy.optimize import linprog
from google.colab import drive

# IMPORTAÇÃO BD_INPUT

bd_input = pd.ExcelFile('/content/Temp_Input_FWSched.xlsx')

df_cad = pd.read_excel(bd_input,'Input_Cadastro', usecols='A:N')
print(df_cad)

df_produt = pd.read_excel(bd_input,'Input_Premissas', usecols='A:C')
df_produt = df_produt.dropna(axis=0, how='all')
print(df_produt)

df_rend = pd.read_excel(bd_input,'Input_Premissas', usecols='E:J')
df_rend = df_rend.dropna(axis=0, how='all')
print(df_rend)

df_sazon = pd.read_excel(bd_input,'Input_Premissas', usecols='L:M')
df_sazon = df_sazon.dropna(axis=0, how='all')
print(df_sazon)

df_horiz = pd.read_excel(bd_input,'Input_Premissas', usecols='O:P')
df_horiz = df_horiz.dropna(axis=0, how='all')
print(df_horiz)

# TABELA DE RENDIMENTOS

# Selecionar quais colunas devem ser concatenadas
col_rend_concat = ['Módulo', 'Máquinas', 'Caixa de Carga (m³)', 'Horas Trabalhadas/dia', 'Learning Curve']

# Criar uma nova coluna 'Agrupamento' com a concatenação das colunas selecionadas
df_rend['Estrato'] = df_rend[col_rend_concat].apply(lambda row: '-'.join(map(str, row)), axis=1)

# Usar o método factorize para atribuir uma nova identificação com base na coluna 'Categoria'
df_rend['ID Estrato'] = pd.factorize(df_rend['Estrato'])[0]

print(df_rend)

# Obter valores únicos da coluna 'Estrato'
matriz_estratos = df_rend['ID Estrato'].unique()

# Criar novos DataFrames para cada valor único e identificar o nome dos DataFrames
for valor in matriz_estratos:
    # Criar o nome do DataFrame usando o valor único
    nome_do_dataframe = f'df_{valor}'

    # Criar o DataFrame específico para o valor único
    locals()[nome_do_dataframe] = df_rend[df_rend['ID Estrato'] == valor].reset_index(drop=True)

    # Exibir o DataFrame e seu nome
    conj_matriz_estr = f"{nome_do_dataframe}:\n{locals()[nome_do_dataframe]}\n"
    print(conj_matriz_estr)

# CADASTRO

# Inclusão colunas dias de eito e pilha
df_cad['Data Corte'] = pd.to_datetime(df_cad['Data Corte'])
data_fixa = pd.to_datetime(df_horiz.iloc[0,1])
df_cad['Dias Eito'] = (data_fixa - df_cad['Data Corte']).dt.days
df_cad['Dias Pilha'] = (df_cad['Data Transporte'] - data_fixa).dt.days
print(df_cad)

# MATRIZ POR ESTRATO - EXEMPLO DF_0

df_0_merged = pd.merge(df_0, df_cad, on='Módulo')

# Converter a coluna 'DataInicial' para o formato datetime
df_0_merged['Data Rend'] = pd.to_datetime(df_0_merged['Data Rend'])

# Adicionar uma coluna com o intervalo de datas para cada mês
df_0_merged['DataMes Rend'] = df_0_merged['Data Rend'].apply(lambda x: pd.date_range(x, periods=pd.Timestamp(x).days_in_month, freq='D'))

# Explodir o DataFrame para que cada data tenha sua própria linha
df_0_exp = df_0_merged.explode('DataMes Rend').reset_index(drop=True)

# Inclusão dos coeficientes no dataframe df_0_exp para cálculo de produtividade
df_0_exp['Coeficiente 1'] = df_0_exp['Produto'].map(df_produt.set_index('Produto')['Coeficiente 1'])
df_0_exp['Coeficiente 2'] = df_0_exp['Produto'].map(df_produt.set_index('Produto')['Coeficiente 2'])

# Cálculo de nº de máquinas por talhão atribuindo premissas de rendimento por estrato
df_0_exp['VolTal Máq'] = df_0_exp['Volume (m³)'] / (((df_0_exp['Distância Baldeio (m)']*df_0_exp['Coeficiente 1']) + df_0_exp['Coeficiente 2']) * df_0_exp['Caixa de Carga (m³)'] * df_0_exp['Learning Curve'] * df_0_exp['Horas Trabalhadas/dia'])

# Equação de otimização para priorização de dias de pilha e alocação de estoque por classe e sazonalidade
df_0_exp['Sazonalidade'] = df_0_exp['Data Rend'].map(df_sazon.set_index('Data Sazon')['Classe Sazon'])

df_0_exp['Delta Eito'] = df_0_exp['DataMes Rend'] - df_0_exp['Data Corte']
df_0_exp['Delta Pilha'] = df_0_exp['Data Transporte'] - df_0_exp['DataMes Rend']

df_0_exp['Float Eito'] = df_0_exp['Delta Eito'].dt.total_seconds() / (24 * 60 * 60)
df_0_exp['Float Pilha'] = df_0_exp['Delta Pilha'].dt.total_seconds() / (24 * 60 * 60)

# Criar a nova coluna 'inicio_mes'
df_0_exp['Mês Transporte'] = df_0_exp['Data Transporte'].dt.to_period('M').dt.to_timestamp()

# Aplicar uma fórmula com base numa condição
df_0_exp['Valor Otm'] = np.where(df_0_exp['Data Rend'] == df_0_exp['Mês Transporte'],
                        (df_0_exp['Float Eito']*10000000) * df_0_exp['Float Pilha'],
                        np.where(((df_0_exp['Sazonalidade'] == 'Chuva') & (df_0_exp['Classe Estoque'] == 'Chuva')) | ((df_0_exp['Sazonalidade'] == 'Seco') & (df_0_exp['Classe Estoque'] == 'Seco')),
                        df_0_exp['Float Eito'] * (df_0_exp['Float Pilha']*1000000),
                        df_0_exp['Float Eito'] * df_0_exp['Float Pilha']))

df_0_exp.to_excel('/content/df_0_exp.xlsx', index=False)
print(df_0_exp)

# Usar pivot para transformar linhas em colunas
df_0_pivot = df_0_exp.pivot(index='Talhão', columns='DataMes Rend', values='Valor Otm').reset_index()
df_0_pivot = df_0_pivot.drop('Talhão', axis=1)

df_0_pivot.to_excel('/content/df_0_pivot.xlsx', index=False)
print(df_0_pivot)

# RESTRIÇÕES POR MATRIZ - EXEMPLO DF_0

# Definição de restrição de volume=máquinas para a lista de talhões da matriz
df_0_restrtal = df_0_exp[['Talhão','VolTal Máq']]
df_0_restrtal = df_0_restrtal.drop_duplicates(subset='Talhão')
print(df_0_restrtal)

# Ordenar os talhões para coincidir com a sequência pivot
df_0_restrtal = df_0_restrtal.sort_values(by=['Talhão'], ascending=[True])
print(df_0_restrtal)

# Definição de restrição de máquinas para o intervalo de dias da matriz
df_0_restrmaq = df_0_exp[['DataMes Rend','Máquinas']]
df_0_restrmaq = df_0_restrmaq.drop_duplicates(subset='DataMes Rend')
print(df_0_restrmaq)

# SOLVER - EXEMPLO DF_0

# CONVERSÃO PARA NUMPY
df_0_bpc_dias = df_0_pivot.to_numpy()
df_0_oferta_tal = df_0_restrtal['VolTal Máq'].to_numpy()
df_0_demanda_maq = df_0_restrmaq['Máquinas'].to_numpy()

print(df_0_bpc_dias)
print(df_0_oferta_tal)
print(df_0_demanda_maq)

# DEFINIÇÃO DOS EIXOS DA MATRIZ DE VARIÁVEIS
df_0_ei = np.ones(df_0_bpc_dias.shape[0])
df_0_ej = np.ones(df_0_bpc_dias.shape[1])

print(df_0_ei.shape)
print(df_0_ej.shape)

# DEFINIÇÃO DAS RESTRIÇÕES
lower = np.zeros(df_0_bpc_dias.shape)
df_0_x = cp.Variable(df_0_bpc_dias.shape)
df_0_bpc0 = df_0_x@df_0_ej <= df_0_oferta_tal
df_0_bpc1 = cp.transpose(df_0_x)@df_0_ei <= df_0_demanda_maq
df_0_bpc2 = df_0_x >= lower

# FUNÇÃO OBJETIVO
df_0_obj = cp.multiply(df_0_bpc_dias,df_0_x)@df_0_ej@df_0_ei

# RESULTADOS DA OTIMIZAÇÃO - EXEMPLO DF_0

# SOLVER
df_0_prob = cp.Problem(cp.Maximize(df_0_obj), [df_0_bpc0, df_0_bpc1, df_0_bpc2])
df_0_solv = df_0_prob.solve(solver=cp.GLPK_MI)

print(df_0_solv)
print("status:",df_0_prob.status)
print('restrição talhão:', np.matmul(df_0_x.value,df_0_ej))
print('restrição máquinas:', np.matmul(np.transpose(df_0_x.value),df_0_ei))
print("Valor ótimo de x:", df_0_x.value)

# EXPORTAÇÃO DOS RESULTADOS - EXEMPLO DF_0

# Criar DataFrama a partir do resultado das variáveis na matriz otm
df_0_mtrz_otm = df_0_x.value
df_0_sched = pd.DataFrame(df_0_mtrz_otm, columns=[f'Coluna_{i}' for i in range(df_0_mtrz_otm.shape[1])])
# print(df_0_sched)

# Criar datas diárias sequenciais a partir de uma data inicial
# Importante aplicar iloc na matriz de rendimento em questão (exemplo: df_0)
df_0_data_i = pd.to_datetime(df_0.iloc[0,5])
df_0_sched_numcol = len(df_0_sched.columns)
df_0_sched_data = [df_0_data_i + datetime.timedelta(days=i) for i in range(df_0_sched_numcol)]
df_0_sched.columns = df_0_sched_data
# print(df_0_sched_data)

# Incluir talhões
df_0_coltal = df_0_restrtal.reset_index()[['Talhão']]
df_0_sched.insert(0, 'Talhão', df_0_coltal)
# print(df_0_coltal)
# print(df_0_sched)

# Exibir o DataFrame resultante
df_0_sched.to_excel('/content/df_0_outmtz.xlsx', index=False)
print(df_0_sched)

# Usar a função melt para transformar colunas em linhas
df_0_sched_transf = pd.melt(df_0_sched, id_vars=['Talhão'], var_name='Data Baldeio', value_name='Máquinas')
df_0_sched = df_0_sched_transf

# filtro de linhas >= 0
df_0_sched = df_0_sched.loc[df_0_sched['Máquinas'] > 0]

# Ordenar o DataFrame com base nas colunas 'Talhão' e 'Data Baldeio' em ordem decrescente
df_0_sched_ord = df_0_sched.sort_values(by=['Talhão', 'Data Baldeio'], ascending=[True, True])
df_0_sched_ord = df_0_sched

# Exibir o DataFrame resultante
df_0_sched.to_excel('/content/df_0_sched.xlsx', index=False)
print(df_0_sched)
