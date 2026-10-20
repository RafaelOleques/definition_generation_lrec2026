import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

RESULTS_DIR = "results"
FOLDS = 5
METRICA = "f1"
IMAGES_DIR = "images"
PROMPT_NAMES = [
    "zero_shot",
    "no_context_no_example",
    "no_context_1_shot",
    "no_context_2_shot",
    "no_context_4_shot",
    "no_context_8_shot",
    "context_1_shot",
    "context_2_shot",
    "context_4_shot",
    "context_8_shot",
]

os.makedirs(IMAGES_DIR, exist_ok=True)

def coletar_dados(results_base_path, prompt_names=PROMPT_NAMES):
    registros = []
    for prompt in prompt_names:
        for fold in range(FOLDS):
            file_path = os.path.join(results_base_path, "iid", f"fold_{fold}", f"{prompt}.json")
            if not os.path.exists(file_path):
                print(f"[AVISO] Arquivo não encontrado: {file_path}")
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                scores = data.get("scores", {}).get("bertscore", {}).get(METRICA, [])
                if not scores:
                    print(f"[AVISO] Scores vazios no arquivo: {file_path}")
                    continue
                media_fold = np.mean(scores) * 100
                registros.append({"prompt": prompt, "fold": fold, "bert_f1": media_fold})
    return pd.DataFrame(registros)

def extrair_shots(prompt):
    if prompt == "no_context_no_example":
        return 0
    elif prompt == "zero_shot":
        return 0
    m = re.search(r"(\d+)_shot", prompt)
    if m:
        return int(m.group(1))
    else:
        return np.nan

def extrair_contexto(prompt):
    if prompt == "zero_shot":
        return "with context"
    elif prompt == "no_context_no_example":
        return "without context"
    elif prompt.startswith("context"):
        return "with context"
    else:
        return "without context"

def gerar_grafico(df, modelo, corpus, min_y, max_y):
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df,
        x="shots",
        y="bert_f1",
        hue="contexto",
        marker="o",
        errorbar="sd"
    )
    #plt.title(f"{modelo} | {corpus}")
    plt.xlabel("Number of shots")
    plt.ylabel("BERTScore F1 (%)")
    plt.ylim(min_y, max_y)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), title="Contexto")
    plt.tight_layout()
    arquivo = os.path.join(IMAGES_DIR, f"{modelo.replace(':','_')}__{corpus}.png")
    plt.savefig(arquivo)
    plt.savefig(arquivo.replace("png", "svg"))
    plt.close()

def gerar_boxplot(df, modelo, corpus, min_y, max_y):
    plt.figure(figsize=(10,6))
    df = df.rename(columns={"contexto": "Context"})

    sns.boxplot(
        data=df,
        x="shots",
        y="bert_f1",
        hue="Context"
    )
    #plt.title(f"Distribuição BERTScore F1 - {modelo} | {corpus}")
    plt.xlabel("Number of shots")
    plt.ylabel("BERTScore F1 (%)")
    plt.ylim(min_y, max_y)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), title="Contexto")
    plt.tight_layout()
    arquivo = os.path.join(IMAGES_DIR, f"boxplot_{modelo.replace(':','_')}__{corpus}.png")
    plt.savefig(arquivo)
    plt.savefig(arquivo.replace("png", "svg"))
    plt.close()

def gerar_grafico_comparativo(df_comparativo, min_y, max_y):
    corpus = df_comparativo["corpus"].iloc[0]
    plt.figure(figsize=(12,6))
    df = df_comparativo.rename(columns={"modelo": "Model", "contexto": "Context"})

    sns.lineplot(
        data=df,
        x="shots",
        y="bert_f1",
        hue="Model",
        style="Context",
        marker="o",
        errorbar="sd"
    )
    #plt.title(f"Comparação entre modelos - {corpus}")
    plt.xlabel("Number of shots")
    plt.ylabel("BERTScore F1 (%)")
    plt.ylim(min_y, max_y)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), title="Model / Context")
    plt.tight_layout()
    arquivo = os.path.join(IMAGES_DIR, f"comparacao_modelos_{corpus}.png")
    plt.savefig(arquivo)
    plt.savefig(arquivo.replace("png", "svg"))
    plt.close()

def gerar_sunburst(df):
    import plotly.express as px

    if "contexto" not in df.columns:
        print("[ERRO] Coluna 'contexto' não encontrada no dataframe.")
        return

    df_agg = df.groupby(['corpus', 'modelo', 'contexto', 'shots'], as_index=False).agg({
        'bert_f1': 'mean',
        'prompt': 'count'
    }).rename(columns={'prompt':'count'})

    fig = px.sunburst(
        df_agg,
        path=['corpus', 'modelo', 'contexto', 'shots'],
        values='bert_f1',
        color='bert_f1',
        color_continuous_scale='Viridis',
        title='Sunburst: Corpus → Modelo → Contexto → Shots (tamanho, cor = F1 médio)'
    )

    fig.update_traces(texttemplate='%{label}', textinfo='label+percent entry')
    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))

    arquivo = os.path.join(IMAGES_DIR, "sunburst_unico.png")
    fig.write_image(arquivo)

def main():
    dados_todos_modelos = []

    modelos = sorted(os.listdir(RESULTS_DIR))
    for modelo in modelos:
        modelo_path = os.path.join(RESULTS_DIR, modelo)
        if not os.path.isdir(modelo_path):
            continue

        corpora = sorted(os.listdir(modelo_path))
        for corpus in corpora:
            corpus_path = os.path.join(modelo_path, corpus)
            if not os.path.isdir(corpus_path):
                continue

            print(f"Lendo dados de: {modelo} / {corpus}")
            df = coletar_dados(corpus_path, PROMPT_NAMES)

            if df.empty:
                print(f"[AVISO] Dados vazios para {modelo} / {corpus}, pulando...")
                continue

            counts = df['prompt'].value_counts()
            prompts_completos = counts[counts == FOLDS].index.tolist()
            df = df[df['prompt'].isin(prompts_completos)]

            if df.empty:
                print(f"[AVISO] Nenhum prompt completo para {modelo} / {corpus}, pulando...")
                continue

            df["shots"] = df["prompt"].apply(extrair_shots)
            df["contexto"] = df["prompt"].apply(extrair_contexto)
            df["modelo"] = modelo
            df["corpus"] = corpus
            dados_todos_modelos.append(df)

    if not dados_todos_modelos:
        print("Nenhum dado para plotar.")
        return

    df_todos = pd.concat(dados_todos_modelos, ignore_index=True)

    min_y = max(0, df_todos["bert_f1"].min() - 5)
    max_y = min(100, df_todos["bert_f1"].max() + 5)

    for df in dados_todos_modelos:
        modelo = df["modelo"].iloc[0]
        corpus = df["corpus"].iloc[0]
        gerar_grafico(df, modelo, corpus, min_y, max_y)
        gerar_boxplot(df, modelo, corpus, min_y, max_y)

    for corpus, df_corpus in df_todos.groupby("corpus"):
        print(f"Gerando gráfico comparativo para corpus: {corpus}")
        gerar_grafico_comparativo(df_corpus, min_y, max_y)

    if not df_todos.empty:
        gerar_sunburst(df_todos)

    print(f"Gráficos salvos em '{IMAGES_DIR}'.")

if __name__ == "__main__":
    main()
