# Tactical Forwarder Planning in Forestry Supply Chain

# importing libraries
import numpy as np
import pandas as pd
import scipy as sp
import cvxpy as cp
import datetime

from scipy.optimize import linprog
from google.colab import drive

# importing database inputs
bd_input = pd.ExcelFile('/content/Temp_Input_FWSched.xlsx')
df_cad = pd.read_excel(bd_input,'Input_Cadastro', usecols='A:N')

df_produt = pd.read_excel(bd_input,'Input_Premissas', usecols='A:C')
df_produt = df_produt.dropna(axis=0, how='all')

df_rend = pd.read_excel(bd_input,'Input_Premissas', usecols='E:J')
df_rend = df_rend.dropna(axis=0, how='all')

df_sazon = pd.read_excel(bd_input,'Input_Premissas', usecols='L:M')
df_sazon = df_sazon.dropna(axis=0, how='all')

df_horiz = pd.read_excel(bd_input,'Input_Premissas', usecols='O:P')
df_horiz = df_horiz.dropna(axis=0, how='all')

# income table
col_rend_concat = ['Módulo', 'Máquinas', 'Caixa de Carga (m³)', 'Horas Trabalhadas/dia', 'Learning Curve']
df_rend['Estrato'] = df_rend[col_rend_concat].apply(lambda row: '-'.join(map(str, row)), axis=1)
df_rend['ID Estrato'] = pd.factorize(df_rend['Estrato'])[0]

matriz_estratos = df_rend['ID Estrato'].unique()
for valor in matriz_estratos:
    nome_do_dataframe = f'df_{valor}'
    locals()[nome_do_dataframe] = df_rend[df_rend['ID Estrato'] == valor].reset_index(drop=True)
    conj_matriz_estr = f"{nome_do_dataframe}:\n{locals()[nome_do_dataframe]}\n"

# forest registry
df_cad['Data Corte'] = pd.to_datetime(df_cad['Data Corte'])
data_fixa = pd.to_datetime(df_horiz.iloc[0,1])
df_cad['Dias Eito'] = (data_fixa - df_cad['Data Corte']).dt.days
df_cad['Dias Pilha'] = (df_cad['Data Transporte'] - data_fixa).dt.days

# matrix by stratum - example DF_0
df_0_merged = pd.merge(df_0, df_cad, on='Módulo')
df_0_merged['Data Rend'] = pd.to_datetime(df_0_merged['Data Rend'])
df_0_merged['DataMes Rend'] = df_0_merged['Data Rend'].apply(lambda x: pd.date_range(x, periods=pd.Timestamp(x).days_in_month, freq='D'))
df_0_exp = df_0_merged.explode('DataMes Rend').reset_index(drop=True)

df_0_exp['Coeficiente 1'] = df_0_exp['Produto'].map(df_produt.set_index('Produto')['Coeficiente 1'])
df_0_exp['Coeficiente 2'] = df_0_exp['Produto'].map(df_produt.set_index('Produto')['Coeficiente 2'])

df_0_exp['VolTal Máq'] = df_0_exp['Volume (m³)'] / (((df_0_exp['Distância Baldeio (m)']*df_0_exp['Coeficiente 1']) + df_0_exp['Coeficiente 2']) * df_0_exp['Caixa de Carga (m³)'] * df_0_exp['Learning Curve'] * df_0_exp['Horas Trabalhadas/dia'])
df_0_exp['Sazonalidade'] = df_0_exp['Data Rend'].map(df_sazon.set_index('Data Sazon')['Classe Sazon'])
df_0_exp['Delta Eito'] = df_0_exp['DataMes Rend'] - df_0_exp['Data Corte']
df_0_exp['Delta Pilha'] = df_0_exp['Data Transporte'] - df_0_exp['DataMes Rend']
df_0_exp['Float Eito'] = df_0_exp['Delta Eito'].dt.total_seconds() / (24 * 60 * 60)
df_0_exp['Float Pilha'] = df_0_exp['Delta Pilha'].dt.total_seconds() / (24 * 60 * 60)
df_0_exp['Mês Transporte'] = df_0_exp['Data Transporte'].dt.to_period('M').dt.to_timestamp()

df_0_exp['Valor Otm'] = np.where(df_0_exp['Data Rend'] == df_0_exp['Mês Transporte'],
                        (df_0_exp['Float Eito']*10000000) * df_0_exp['Float Pilha'],
                        np.where(((df_0_exp['Sazonalidade'] == 'Chuva') & (df_0_exp['Classe Estoque'] == 'Chuva')) | ((df_0_exp['Sazonalidade'] == 'Seco') & (df_0_exp['Classe Estoque'] == 'Seco')),
                        df_0_exp['Float Eito'] * (df_0_exp['Float Pilha']*1000000),
                        df_0_exp['Float Eito'] * df_0_exp['Float Pilha']))

df_0_exp.to_excel('/content/df_0_exp.xlsx', index=False)

df_0_pivot = df_0_exp.pivot(index='Talhão', columns='DataMes Rend', values='Valor Otm').reset_index()
df_0_pivot = df_0_pivot.drop('Talhão', axis=1)
df_0_pivot.to_excel('/content/df_0_pivot.xlsx', index=False)

# restrictions by matrix - example DF_0
df_0_restrtal = df_0_exp[['Talhão','VolTal Máq']]
df_0_restrtal = df_0_restrtal.drop_duplicates(subset='Talhão')
df_0_restrtal = df_0_restrtal.sort_values(by=['Talhão'], ascending=[True])
df_0_restrmaq = df_0_exp[['DataMes Rend','Máquinas']]
df_0_restrmaq = df_0_restrmaq.drop_duplicates(subset='DataMes Rend')

# solver - example DF_0
df_0_bpc_dias = df_0_pivot.to_numpy()
df_0_oferta_tal = df_0_restrtal['VolTal Máq'].to_numpy()
df_0_demanda_maq = df_0_restrmaq['Máquinas'].to_numpy()

df_0_ei = np.ones(df_0_bpc_dias.shape[0])
df_0_ej = np.ones(df_0_bpc_dias.shape[1])

lower = np.zeros(df_0_bpc_dias.shape)
df_0_x = cp.Variable(df_0_bpc_dias.shape)
df_0_bpc0 = df_0_x@df_0_ej <= df_0_oferta_tal
df_0_bpc1 = cp.transpose(df_0_x)@df_0_ei <= df_0_demanda_maq
df_0_bpc2 = df_0_x >= lower

df_0_obj = cp.multiply(df_0_bpc_dias,df_0_x)@df_0_ej@df_0_ei

# optimization results - example DF_0
df_0_prob = cp.Problem(cp.Maximize(df_0_obj), [df_0_bpc0, df_0_bpc1, df_0_bpc2])
df_0_solv = df_0_prob.solve(solver=cp.GLPK_MI)

print(df_0_solv)
print("status:",df_0_prob.status)
print('restrição talhão:', np.matmul(df_0_x.value,df_0_ej))
print('restrição máquinas:', np.matmul(np.transpose(df_0_x.value),df_0_ei))
print("Valor ótimo de x:", df_0_x.value)

# exporting results - example DF_0
df_0_mtrz_otm = df_0_x.value
df_0_sched = pd.DataFrame(df_0_mtrz_otm, columns=[f'Coluna_{i}' for i in range(df_0_mtrz_otm.shape[1])])

df_0_data_i = pd.to_datetime(df_0.iloc[0,5])
df_0_sched_numcol = len(df_0_sched.columns)
df_0_sched_data = [df_0_data_i + datetime.timedelta(days=i) for i in range(df_0_sched_numcol)]
df_0_sched.columns = df_0_sched_data

df_0_coltal = df_0_restrtal.reset_index()[['Talhão']]
df_0_sched.insert(0, 'Talhão', df_0_coltal)

df_0_sched.to_excel('/content/df_0_outmtz.xlsx', index=False)
df_0_sched_transf = pd.melt(df_0_sched, id_vars=['Talhão'], var_name='Data Baldeio', value_name='Máquinas')
df_0_sched = df_0_sched_transf

df_0_sched = df_0_sched.loc[df_0_sched['Máquinas'] > 0]
df_0_sched_ord = df_0_sched.sort_values(by=['Talhão', 'Data Baldeio'], ascending=[True, True])
df_0_sched_ord = df_0_sched
df_0_sched.to_excel('/content/df_0_sched.xlsx', index=False)
