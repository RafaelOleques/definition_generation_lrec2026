import os
import json
import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene, wilcoxon, friedmanchisquare
import scikit_posthocs as sp
import statsmodels.stats.multitest as smm

# --- 1. Coleta dos dados reais dos JSONs ---
def coletar_dados(results_base_path, prompt_names):
    registros = []
    for prompt in prompt_names:
        for fold in range(5):  # 5 folds
            file_path = os.path.join(results_base_path, f"fold_{fold}", f"{prompt}.json")
            if not os.path.exists(file_path):
                print(f"[AVISO] Arquivo não encontrado: {file_path}")
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                scores = data["scores"]["bertscore"]["f1"]
                media_fold = np.mean(scores) * 100  # transformar em %
                registros.append({"prompt": prompt, "fold": fold, "bert_f1": media_fold})
    return pd.DataFrame(registros)

# --- 2. Extrair número de shots e contexto do nome do prompt ---
def extrair_shots(prompt):
    import re
    m = re.search(r"(\d+)_shot", prompt)
    if m:
        return int(m.group(1))
    elif prompt in ["zero_shot", "no_context_no_example"]:
        return 0
    else:
        return np.nan

def extrair_contexto(prompt):
    if prompt == "zero_shot":
        return "com contexto"
    elif prompt == "no_context_no_example":
        return "sem contexto"
    elif prompt.startswith("context"):
        return "com contexto"
    else:
        return "sem contexto"

# --- 3. Função para montar os dados agregados para análise (dicionários para cada contexto com arrays por shots) ---
def organizar_dados(df):
    dados_com = {}
    dados_sem = {}
    for shots in sorted(df["shots"].unique()):
        vals_com = df[(df["shots"]==shots) & (df["contexto"]=="com contexto")]["bert_f1"].values
        vals_sem = df[(df["shots"]==shots) & (df["contexto"]=="sem contexto")]["bert_f1"].values
        if len(vals_com) > 0:
            dados_com[shots] = vals_com
        if len(vals_sem) > 0:
            dados_sem[shots] = vals_sem
    return dados_com, dados_sem

# --- 4. Funções de resumo e testes baseados no exemplo que você deu ---

def resumo(data_dict):
    resumo_lista = []
    for shots_, vals in sorted(data_dict.items()):
        resumo_lista.append({
            "Número de exemplos (shots)": shots_,
            "Média": np.mean(vals),
            "Desvio": np.std(vals, ddof=1)
        })
    return pd.DataFrame(resumo_lista).sort_values("Número de exemplos (shots)")

def testar_normalidade(data_dict):
    res = []
    for shots_, vals in sorted(data_dict.items()):
        stat, p = shapiro(vals)
        res.append({"Número de exemplos (shots)": shots_, "Shapiro p-valor": p})
    return pd.DataFrame(res).set_index("Número de exemplos (shots)")

def testar_levene(dados1, dados2):
    res = []
    for shots_ in sorted(set(dados1.keys()) & set(dados2.keys())):
        stat, p = levene(dados1[shots_], dados2[shots_])
        res.append({"Número de exemplos (shots)": shots_, "Levene p-valor": p})
    return pd.DataFrame(res).set_index("Número de exemplos (shots)")

def testar_wilcoxon(dados1, dados2):
    res = []
    for shots_ in sorted(set(dados1.keys()) & set(dados2.keys())):
        try:
            stat, p = wilcoxon(dados1[shots_], dados2[shots_])
        except ValueError:
            p = np.nan
        res.append({"Número de exemplos (shots)": shots_, "Wilcoxon p-valor": p})
    return pd.DataFrame(res).set_index("Número de exemplos (shots)")

def testar_friedman(data_dict):
    arrays = [data_dict[s] for s in sorted(data_dict.keys())]
    stat, p = friedmanchisquare(*arrays)
    return stat, p

def pos_testes(data_dict, nome_contexto, alpha=0.05):
    grupos_ordenados = sorted(data_dict.keys())
    dados_wide = np.column_stack([data_dict[shots_] for shots_ in grupos_ordenados])

    print(f"\nPós-testes para contexto '{nome_contexto}':\n")

    data_long = []
    groups_long = []
    for shots_ in grupos_ordenados:
        data_long.extend(data_dict[shots_])
        groups_long.extend([shots_] * len(data_dict[shots_]))

    df_long = pd.DataFrame({"value": data_long, "group": groups_long})

    def print_significant_pairs(df_pvals, method_name):
        print(f"{method_name} — pares com diferença significativa (p < {alpha}):")
        significant = False
        for i, group1 in enumerate(df_pvals.index):
            for j, group2 in enumerate(df_pvals.columns):
                if i < j:  # só metade superior da matriz
                    pval = df_pvals.iloc[i, j]
                    if pval < alpha:
                        significant = True
                        print(f"- {group1} vs {group2}: p = {pval:.4f}")
        if not significant:
            print("Nenhum par com diferença significativa encontrado.")
        print()

    # Dunn test
    dunn_res = sp.posthoc_dunn(df_long, val_col="value", group_col="group", p_adjust='bonferroni')
    print_significant_pairs(dunn_res, "Dunn test (Bonferroni)")

    # Nemenyi test (matriz 2D)
    nemenyi_res = sp.posthoc_nemenyi_friedman(dados_wide)
    nemenyi_df = pd.DataFrame(nemenyi_res, index=grupos_ordenados, columns=grupos_ordenados)
    print_significant_pairs(nemenyi_df, "Nemenyi test")

    # Conover test
    conover_res = sp.posthoc_conover(df_long, val_col="value", group_col="group", p_adjust='holm')
    print_significant_pairs(conover_res, "Conover test (Holm)")

# --- 5. Funções para formatação e impressão ---

def format_perf(mean, std):
    return f"{mean:.2f} ± {std:.2f}"

def format_pval(p):
    if p is None or np.isnan(p):
        return "-"
    elif p < 0.001:
        return "<0.001"
    else:
        return f"{p:.4f}"

def print_desempenho(df_perf):
    print("### Desempenho médio (%) ± desvio padrão por número de exemplos e contexto\n")
    df_pivot = df_perf.pivot(index="Número de exemplos (shots)", columns="Contexto", values="Média")
    df_std = df_perf.pivot(index="Número de exemplos (shots)", columns="Contexto", values="Desvio")
    print(f"{'Shots':>6} | {'Com contexto':^20} | {'Sem contexto':^20}")
    print("-"*55)
    for shots in df_pivot.index:
        cmean = df_pivot.loc[shots, 'com contexto'] if 'com contexto' in df_pivot.columns else np.nan
        cstd = df_std.loc[shots, 'com contexto'] if 'com contexto' in df_std.columns else np.nan
        scmean = df_pivot.loc[shots, 'sem contexto'] if 'sem contexto' in df_pivot.columns else np.nan
        scstd = df_std.loc[shots, 'sem contexto'] if 'sem contexto' in df_std.columns else np.nan
        print(f"{shots:6} | {format_perf(cmean, cstd):^20} | {format_perf(scmean, scstd):^20}")
    print("\nLegenda: Média ± Desvio padrão (%)\n")

def print_teste_normalidade_homogeneidade(shapiro_com, shapiro_sem, levene_res):
    print("### Testes de normalidade (Shapiro) e homogeneidade de variância (Levene)\n")
    print(f"{'Shots':>6} | {'Shapiro (com contexto)':^22} | {'Shapiro (sem contexto)':^22} | {'Levene (p-valor)':^17}")
    print("-"*75)
    for shots in sorted(set(shapiro_com.index) | set(shapiro_sem.index)):
        sc = format_pval(shapiro_com.loc[shots, "Shapiro p-valor"]) if shots in shapiro_com.index else "-"
        ss = format_pval(shapiro_sem.loc[shots, "Shapiro p-valor"]) if shots in shapiro_sem.index else "-"
        lev = format_pval(levene_res.loc[shots, "Levene p-valor"]) if shots in levene_res.index else "-"
        print(f"{shots:6} | {sc:^22} | {ss:^22} | {lev:^17}")
    print("\nLegenda: p-valor; < 0.05 indica rejeição da hipótese nula.\n")

def print_teste_wilcoxon(wilcoxon_res):
    print("### Teste de Wilcoxon pareado entre contextos por número de exemplos\n")
    print(f"{'Shots':>6} | {'p-valor Wilcoxon':^17} | {'Diferença média significativa?':^28}")
    print("-"*55)
    for shots in sorted(wilcoxon_res.index):
        pval = wilcoxon_res.loc[shots, "Wilcoxon p-valor"]
        signif = "Sim" if pval < 0.05 else "Não"
        print(f"{shots:6} | {format_pval(pval):^17} | {signif:^28}")
    print()

def print_friedman(friedman_com, friedman_sem):
    print("### Teste de Friedman para variação do desempenho por número de exemplos dentro de cada contexto\n")
    for contexto, res in zip(["com contexto", "sem contexto"], [friedman_com, friedman_sem]):
        stat, p = res
        signif = "Sim" if p < 0.05 else "Não"
        print(f"Contexto '{contexto}': estatística = {stat:.4f}, p-valor = {format_pval(p)} → Diferença significativa? {signif}\n")

# --- 6. Execução completa ---

def main():
    prompt_names = [
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

    results_base_path = "results/gemma3:12b/definitions_geocorpus/iid"  # ajuste o caminho conforme seu ambiente

    # Coletar dados
    df = coletar_dados(results_base_path, prompt_names)
    df["shots"] = df["prompt"].apply(extrair_shots)
    df["contexto"] = df["prompt"].apply(extrair_contexto)

    # Organizar dados por contexto e shots
    dados_com, dados_sem = organizar_dados(df)

    # Resumos
    resumo_com = resumo(dados_com)
    resumo_com["Contexto"] = "com contexto"
    resumo_sem = resumo(dados_sem)
    resumo_sem["Contexto"] = "sem contexto"
    resumo_total = pd.concat([resumo_com, resumo_sem])

    print_desempenho(resumo_total)

    # Testes normalidade e homogeneidade
    shapiro_com = testar_normalidade(dados_com)
    shapiro_sem = testar_normalidade(dados_sem)
    levene_res = testar_levene(dados_com, dados_sem)
    print_teste_normalidade_homogeneidade(shapiro_com, shapiro_sem, levene_res)

    # Teste Wilcoxon entre contextos
    wilcoxon_res = testar_wilcoxon(dados_com, dados_sem)
    print_teste_wilcoxon(wilcoxon_res)

    # Teste Friedman dentro de cada contexto
    friedman_com = testar_friedman(dados_com)
    friedman_sem = testar_friedman(dados_sem)
    print_friedman(friedman_com, friedman_sem)

    # Se Friedman significativo, pós-testes
    if friedman_com[1] < 0.05:
        pos_testes(dados_com, "com contexto")
    if friedman_sem[1] < 0.05:
        pos_testes(dados_sem, "sem contexto")

if __name__ == "__main__":
    main()
