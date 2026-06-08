# -*- coding: utf-8 -*-
"""
Mini-Projeto Avaliativo - Módulo 1 - Semana 08
Ajustado para execução local no VS Code.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import re

# Configuração para evitar problemas de exibição de caracteres nos gráficos do VS Code
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. FUNÇÕES UTILITÁRIAS E DE FORMATAÇÃO
# ==========================================

def format_brl(value, decimals=2):
    """Formatador customizado para o padrão monetário brasileiro."""
    if pd.isna(value):
        return "R$ 0,00"
    formatted_str = f"{value:,.{decimals}f}"
    temp = formatted_str.replace('.', '#DECIMAL#')
    temp = temp.replace(',', '.')
    final_str = temp.replace('#DECIMAL#', ',')
    return f"R$ {final_str}"

def validar_cliente(nome):
    """RF13 - Validação de integridade de strings via Expressões Regulares."""
    padrao = r"^Cliente_\d{3}$"
    return bool(re.match(padrao, nome))

# ==========================================
# 2. PROCESSAMENTO DE DADOS (FUNÇÕES GLOBAIS)
# ==========================================

def gerar_dataset_vendas(n_registros=200, seed=42):
    """RF01 - Criação do Dataset de Vendas Sintético com anomalias intencionais."""
    random.seed(seed)
    np.random.seed(seed)

    produtos = ["Notebook", "Smartphone", "Tablet", "Monitor", "Teclado", "Mouse", "Headset"]
    categorias = {
        "Notebook": "Computadores", "Smartphone": "Celulares", "Tablet": "Celulares",
        "Monitor": "Computadores", "Teclado": "Periféricos", "Mouse": "Periféricos", "Headset": "Periféricos"
    }
    regioes = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
    clientes = [f"Cliente_{i:03d}" for i in range(1, 51)]
    data_inicio = datetime(2024, 1, 1)
    dados = []

    for i in range(n_registros):
        produto = random.choice(produtos)
        quantidade = random.randint(1, 10)
        preco_base = {"Notebook": 3500, "Smartphone": 2200, "Tablet": 1800, "Monitor": 1200, "Teclado": 250, "Mouse": 120, "Headset": 350}[produto]
        preco = round(preco_base * random.uniform(0.85, 1.15), 2)
        data = data_inicio + timedelta(days=random.randint(0, 364))

        # Erros propositadamente inseridos
        if random.random() < 0.05: quantidade = None
        if random.random() < 0.04: preco = None
        if random.random() < 0.03: produto = " " + produto

        dados.append({
            "id_venda": i + 1,
            "data_venda": data.strftime("%Y-%m-%d"),
            "cliente": random.choice(clientes),
            "produto": produto,
            "categoria": categorias.get(produto.strip(), "Outros"),
            "regiao": random.choice(regioes),
            "quantidade": quantidade,
            "preco_unitario": preco
        })
    return pd.DataFrame(dados)

def inspecionar_dados(df):
    """RF02 - Inspeção estrutural e descritiva inicial."""
    print("\n--- Shape do Dataset ---")
    print(df.shape)
    print("\n--- Colunas Disponíveis ---")
    print(df.columns.tolist())
    print("\n--- Tipos de Dados ---")
    print(df.dtypes)
    print("\n--- Valores Nulos por Coluna ---")
    print(df.isnull().sum())
    print("\n--- Primeiras Linhas ---")
    print(df.head())
    print("\n--- Estatísticas Descritivas ---")
    print(df.describe())

def limpar_dados(df):
    """RF03 - Tratamento, preenchimento e coerção de tipos."""
    n_inicial = len(df)
    relatorio = {}

    # 1. Remover espaços extras
    colunas_texto = df.select_dtypes(include="object").columns
    for col in colunas_texto:
        df[col] = df[col].str.strip()

    # 2. Converter data e descartar anomalias
    df["data_venda"] = pd.to_datetime(df["data_venda"], errors="coerce")
    n_datas_invalidas = df["data_venda"].isnull().sum()
    df = df.dropna(subset=["data_venda"])
    relatorio["datas_invalidas_removidas"] = n_datas_invalidas

    # 3. Eliminar nulos críticos
    n_antes_nulos = len(df)
    df = df.dropna(subset=["quantidade", "preco_unitario"])
    relatorio["linhas_nulas_removidas"] = n_antes_nulos - len(df)

    # 4. Ajustar tipagem estrita
    df["quantidade"] = df["quantidade"].astype(int)
    df["preco_unitario"] = df["preco_unitario"].astype(float)

    relatorio["registros_iniciais"] = n_inicial
    relatorio["registros_finais"] = len(df)
    relatorio["registros_removidos_total"] = n_inicial - len(df)

    return df, relatorio

def criar_colunas_derivadas(df):
    """RF04 - Engenharia de Atributos Temporais e Classificação Multicondicional NumPy."""
    df["receita_total"] = df["quantidade"] * df["preco_unitario"]
    df["mes"] = df["data_venda"].dt.month
    df["mes_nome"] = df["data_venda"].dt.strftime("%B")
    df["trimestre"] = df["data_venda"].dt.quarter.apply(lambda q: f"Q{q}")
    df["ano"] = df["data_venda"].dt.year

    condicoes = [
        df["receita_total"] < 500,
        (df["receita_total"] >= 500) & (df["receita_total"] < 5000),
        df["receita_total"] >= 5000
    ]
    classificacoes = ["Baixo Valor", "Médio Valor", "Alto Valor"]
    df["faixa_receita_item"] = np.select(condicoes, classificacoes, default="Não Classificado")
    return df

def calcular_metricas(df):
    """RF05 - Agrupamentos e Métricas Agregadas Corporativas."""
    metricas = {}
    metricas['por_mes'] = df.groupby("mes").agg(
        receita_total=("receita_total", "sum"),
        quantidade=("quantidade", "sum")
    )
    metricas['top_produtos'] = df.groupby("produto")["receita_total"].sum().sort_values(ascending=False).head(5)
    metricas['por_categoria'] = df.groupby("categoria")["receita_total"].sum().sort_values(ascending=False)
    metricas['por_regiao'] = df.groupby("regiao")["receita_total"].sum().sort_values(ascending=False)
    return metricas

def analisar_estatisticas_numpy(df):
    """RF07 - Operações Estatísticas Vetorizadas Nativas via NumPy Array."""
    valores_receita = df["receita_total"].to_numpy()
    return {
        "media": np.mean(valores_receita),
        "mediana": np.median(valores_receita),
        "desvio_padrao": np.std(valores_receita),
        "receita_maxima": np.max(valores_receita),
        "receita_minima": np.min(valores_receita)
    }

# ==========================================
# 3. ARQUITETURA ORIENTADA A OBJETOS (POO)
# ==========================================

class PipelineVendas:
    """RF09 - Orquestrador de Processos de Engenharia de Dados."""
    def __init__(self, caminho_csv=None):
        self.caminho_csv = caminho_csv
        self.df = None
        self.relatorio_limpeza = None

    def carregar_dados(self):
        if self.caminho_csv and os.path.exists(self.caminho_csv):
            self.df = pd.read_csv(self.caminho_csv, encoding='utf-8')
        else:
            self.df = gerar_dataset_vendas()
        return self.df

    def limpar_dados(self):
        self.df, self.relatorio_limpeza = limpar_dados(self.df)
        return self.df

    def mapear_produto_categoria(self):
        if self.df is not None:
            self.df = criar_colunas_derivadas(self.df)
        return self.df

    def executar_pipeline(self):
        print("\n[Início] Executando Pipeline Estruturado...")
        self.carregar_dados()
        self.limpar_dados()
        self.mapear_produto_categoria()
        print("[Sucesso] Pipeline de Vendas concluído.")
        return self.df

class PipelineAvancado(PipelineVendas):
    """RF10 - Extensão do Orquestrador Base aplicando Herança de POO."""
    def __init__(self, caminho_csv=None):
        super().__init__(caminho_csv=caminho_csv)

    def obter_segmentacao_clientes(self):
        """RF06 - Agrupamento e Segmentação de Clientes por Faixa de Valor."""
        if self.df is None:
            return None
        gastos = self.df.groupby("cliente")["receita_total"].sum().reset_index()
        gastos.columns = ["cliente", "total_gasto"]
        gastos["segmento"] = gastos["total_gasto"].apply(
            lambda x: "Ouro" if x > 15000 else ("Prata" if x >= 5000 else "Bronze")
        )
        return gastos

    def receita_por_regiao(self):
        if self.df is None:
            self.executar_pipeline()
        return self.df.groupby("regiao")["receita_total"].sum().sort_values(ascending=False)

# ==========================================
# 4. EXPORTAÇÃO E PLOTAGEM DE GRÁFICOS (RF08)
# ==========================================

def gerar_salvar_graficos(df):
    """RF08 - Gera e exporta visões de negócios em arquivos físicos (PNG)."""
    # Gráfico 1 - Receita por Região
    plt.figure(figsize=(8, 5))
    df.groupby("regiao")["receita_total"].sum().plot(kind="bar", color="skyblue")
    plt.title("Receita por Região (em Reais)", fontweight="bold")
    plt.ylabel("Receita Total")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("grafico_receita_regiao.png", dpi=300)
    plt.close()

    # Gráfico 2 - Top Produtos
    plt.figure(figsize=(10, 5))
    df.groupby("produto")["receita_total"].sum().sort_values(ascending=False).head(5).plot(kind="bar", color="salmon")
    plt.title("Receita por Produto (em Reais)", fontweight="bold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("grafico_top_produtos.png", dpi=300)
    plt.close()

    # Gráfico 3 - Histograma
    plt.figure(figsize=(8, 5))
    plt.hist(df["receita_total"], bins=20, color="lightgreen", edgecolor="black")
    plt.title("Distribuição das Receitas", fontweight="bold")
    plt.xlabel("Receita Total")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig("grafico_distribuicao_receitas.png", dpi=300)
    plt.close()

    # Gráfico 4 - Boxplot Categoria x Receita
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="categoria", y="receita_total", data=df, hue="categoria", legend=False, palette="Set2")
    plt.xticks(rotation=20)
    plt.title("Dispersão de Receita por Categoria", fontweight="bold")
    plt.tight_layout()
    plt.savefig("grafico_boxplot_categorias.png", dpi=300)
    plt.close()
    print("\n[Gráficos] Todos os 4 gráficos exigidos foram guardados localmente (.png)!")

# ==========================================
# 5. FLUXO CENTRALIZADO (FUNÇÃO MAIN)
# ==========================================

def main(csv_path=None):
    # Inicialização unificada do Pipeline Avançado (Herança)
    pipeline = PipelineAvancado(csv_path)
    df_processado = pipeline.executar_pipeline()

    # RF02 - Executar Inspeção Inicial controlada
    print("\n" + "="*40)
    print("=== INSPEÇÃO DE DADOS ===")
    print("="*40)
    inspecionar_dados(df_processado)

    # Exibir Relatório de Limpeza Estruturado (RF03)
    print("\n" + "="*40)
    print("=== RELATÓRIO DE LIMPEZA ===")
    print("="*40)
    for chave, valor in pipeline.relatorio_limpeza.items():
        print(f"{chave.replace('_', ' ').upper()}: {valor}")

    # RF05 - Exibição de Métricas Agregadas conforme padrão exigido
    metricas = calcular_metricas(df_processado)

    print("\n" + "="*40)
    print("=== RECEITA POR MÊS ===")
    print("="*40)
    df_mes_fmt = metricas['por_mes'].copy()
    df_mes_fmt['receita_total'] = df_mes_fmt['receita_total'].apply(format_brl)
    print(df_mes_fmt)

    print("\n" + "="*40)
    print("=== TOP 5 PRODUTOS POR RECEITA ===")
    print("="*40)
    print(metricas['top_produtos'].apply(format_brl))

    print("\n" + "="*40)
    print("=== RECEITA POR CATEGORIA ===")
    print("="*40)
    print(metricas['por_categoria'].apply(format_brl))

    print("\n" + "="*40)
    print("=== RECEITA POR REGIÃO ===")
    print("="*40)
    print(metricas['por_regiao'].apply(format_brl))

    # RF06 - Análise Estendida de Segmentos de Clientes
    print("\n" + "="*40)
    print("=== RESUMO DE SEGMENTAÇÃO DE CLIENTES ===")
    print("="*40)
    clientes_seg = pipeline.obter_segmentacao_clientes()
    resumo_segmentos = clientes_seg.groupby('segmento').agg(
        qtd_clientes=('cliente', 'count'),
        media_gasto=('total_gasto', 'mean')
    ).reset_index()
    resumo_segmentos['media_gasto'] = resumo_segmentos['media_gasto'].apply(format_brl)
    print(resumo_segmentos.to_string(index=False))

    # RF07 - Impressão NumPy Estatístico Vetorizado
    stats = analisar_estatisticas_numpy(df_processado)
    print("\n" + "="*40)
    print("=== ESTATÍSTICAS NUMPY VETORIZADAS ===")
    print("="*40)
    for chave, valor in stats.items():
        print(f"{chave.upper()}: {format_brl(valor)}")

    # RF11 - Funções de Ordem Superior (FOS) com Lambdas
    aplicar_desconto = lambda valor: valor * 0.90
    valores_exemplo = [100, 200, 300, 400]
    print(f"\n[FOS] Map/Lambda Exemplo (Valores com 10% Desconto): {list(map(aplicar_desconto, valores_exemplo))}")
    altas_receitas = list(filter(lambda x: x > 1000, df_processado["receita_total"]))
    print(f"[FOS] Filter/Lambda Exemplo (Transações > R$ 1.000,00): {len(altas_receitas)} registos.")

    # RF13 - Testes Regex de Validação de Clientes
    print(f"\n[Regex] Teste 'Cliente_001': {'VÁLIDO' if validar_cliente('Cliente_001') else 'INVÁLIDO'}")
    print(f"[Regex] Teste 'Cristiano': {'VÁLIDO' if validar_cliente('Cristiano') else 'INVÁLIDO'}")

    # RF08 - Geração Automática dos Gráficos em Imagens Físicas (Sem travar a execução)
    gerar_salvar_graficos(df_processado)

    # RF12 - Exportação e Persistência de Ficheiros I/O (CSV e JSON) com encoding UTF-8 explícito
    df_processado.to_csv("vendas_tratadas.csv", index=False, encoding='utf-8')
    df_processado.to_json("vendas_tratadas.json", orient="records", indent=4, force_ascii=False)
    clientes_seg.to_csv("clientes_segmentados.csv", index=False, encoding='utf-8')
    
    print("\n[I/O Export] Arquivos 'vendas_tratadas.csv', 'clientes_segmentados.csv' e 'vendas_tratadas.json' criados com sucesso!")
    print("[Ambiente Local] Execução concluída. Todos os relatórios e imagens foram salvos no seu diretório atual.")

# ==========================================
# 6. PONTO DE ENTRADA DO SCRIPT
# ==========================================
if __name__ == "__main__":
    # Garante a criação estruturada do arquivo inicial "vendas.csv" se não existir
    if not os.path.exists("vendas.csv"):
        df_base = gerar_dataset_vendas()
        df_base.to_csv("vendas.csv", index=False, encoding='utf-8')

    # Dispara a rotina centralizada
    main(csv_path="vendas.csv")