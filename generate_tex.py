import os
import json
import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene, wilcoxon, friedmanchisquare, rankdata, norm
import scikit_posthocs as sp
import re
from itertools import combinations
from statsmodels.stats.multitest import multipletests


# Cria pastas principais
os.makedirs("tex/por_modelo", exist_ok=True)
os.makedirs("tex/comparacao_modelos", exist_ok=True)

prompts = [
    "zero_shot", "no_context_no_example",
    "no_context_1_shot", "no_context_2_shot", "no_context_4_shot", "no_context_8_shot",
    "context_1_shot", "context_2_shot", "context_4_shot", "context_8_shot"
]

base = "results"
models = ["mistral:7b", "phi3:14b", "llama3.1:8b", "gemma3:12b", "qwen3:14b", "gaia:4b"]
corpora = ["definitions_geocorpus", "definitions_bete"]
splits = ["iid", "cluster", "adversarial"]

def salvar(df, fname, caption, label, path="tex/por_modelo", idx=True):
    os.makedirs(path, exist_ok=True)
    df.to_latex(
        os.path.join(path, fname + ".tex"),
        index=idx, caption=caption, label=label, float_format="%.4f"
    )

def salvar_comp(df, fname, caption, label, corpus, split, idx=True):
    path = os.path.join("tex", "comparacao_modelos", corpus, split)
    os.makedirs(path, exist_ok=True)
    df.to_latex(
        os.path.join(path, fname + ".tex"),
        index=idx, caption=caption, label=label, float_format="%.4f"
    )

def shots_of(prompt):
    # Zero shots para 'zero_shot' e 'no_context_no_example'
    if prompt in ["zero_shot", "no_context_no_example"]:
        return 0
    m = re.search(r"(\d+)_shot", prompt)
    return int(m.group(1)) if m else 0

def contexto_of(prompt):
    # zero_shot é com contexto; no_context_no_example é sem contexto
    if prompt == "zero_shot" or prompt.startswith("context"):
        return "com contexto"
    else:
        return "sem contexto"

# Parte 1: análise individual por modelo

for model in models:
    for corpus in corpora:
        for split in splits:
            path_base = os.path.join(base, model, corpus, split)
            df_rows = []

            for fold in range(5):
                dpath = os.path.join(path_base, f"fold_{fold}")
                if not os.path.isdir(dpath):
                    continue
                for p in prompts:
                    fpath = os.path.join(dpath, p + ".json")
                    if not os.path.isfile(fpath):
                        continue
                    with open(fpath, encoding="utf-8") as f:
                        j = json.load(f)
                    f1 = np.mean(j["scores"]["bertscore"]["f1"]) * 100
                    df_rows.append({
                        "prompt": p,
                        "shots": shots_of(p),
                        "contexto": contexto_of(p),
                        "fold": fold,
                        "bert_f1": f1
                    })

            df = pd.DataFrame(df_rows)
            if df.empty:
                continue

            # Agrupa somente prompts completos (com 5 folds)
            counts = df['prompt'].value_counts()
            completos = counts[counts == 5].index
            df = df[df['prompt'].isin(completos)]

            # Estatísticas descritivas
            agg = df.groupby(["contexto", "shots"])["bert_f1"].agg(['mean', 'std']).reset_index()
            agg.columns = ["Contexto", "Shots", "Média", "Desvio"]
            salvar(
                agg,
                f"{model}_{corpus}_{split}_descritivo",
                f"Descritivo BERT-F1 para {model} / {corpus}/{split}",
                f"tab:{model}_{corpus}_{split}_desc",
                path=os.path.join("tex", "por_modelo", model.replace(":", "_"))
            )

            # Teste de normalidade (Shapiro) por contexto e shots
            shapiro_results = {}
            for c in ["com contexto", "sem contexto"]:
                grupos = df[df.contexto == c].groupby("shots")["bert_f1"].apply(list).to_dict()
                shapiro_results[c] = [shapiro(grupos[k])[1] if len(grupos[k]) >= 3 else np.nan for k in sorted(grupos.keys())]

            # Teste Levene para homogeneidade entre contextos, shot a shot
            levene_pvalues = []
            shots_sorted = sorted(df.shots.unique())
            for shot in shots_sorted:
                vals_c = df[(df.contexto == "com contexto") & (df.shots == shot)]["bert_f1"]
                vals_s = df[(df.contexto == "sem contexto") & (df.shots == shot)]["bert_f1"]
                if len(vals_c) >= 2 and len(vals_s) >= 2:
                    pval = levene(vals_c, vals_s)[1]
                else:
                    pval = np.nan
                levene_pvalues.append(pval)

            dfnorm = pd.DataFrame({
                "Shots": shots_sorted,
                "Shapiro com contexto": shapiro_results["com contexto"],
                "Shapiro sem contexto": shapiro_results["sem contexto"],
                "Levene": levene_pvalues
            })

            salvar(
                dfnorm,
                f"{model}_{corpus}_{split}_normal_levene",
                f"Normalidade e homogeneidade para {model} / {corpus}/{split}",
                f"tab:{model}_{corpus}_{split}_normlev",
                path=os.path.join("tex", "por_modelo", model.replace(":", "_"))
            )

            # Wilcoxon entre contextos para o mesmo número de shots
            wilcoxon_results = []
            for s in shots_sorted:
                a = df[(df.contexto == "com contexto") & (df.shots == s)].sort_values("fold")["bert_f1"].values
                b = df[(df.contexto == "sem contexto") & (df.shots == s)].sort_values("fold")["bert_f1"].values
                # Só se os dois grupos tiverem dados e o mesmo número de folds
                if len(a) == len(b) and len(a) > 0:
                    # Wilcoxon pareado exige no mínimo 2 amostras
                    if len(a) >= 2:
                        p = wilcoxon(a, b)[1]
                        wilcoxon_results.append([s, p, "Sim" if p < 0.05 else "Não"])
                    else:
                        wilcoxon_results.append([s, np.nan, "N/A"])
                else:
                    wilcoxon_results.append([s, np.nan, "N/A"])

            dfwil = pd.DataFrame(wilcoxon_results, columns=["Shots", "p-value", "Significativo"])
            salvar(
                dfwil,
                f"{model}_{corpus}_{split}_wilcoxon",
                f"Wilcoxon entre contextos para {model} / {corpus}/{split}",
                f"tab:{model}_{corpus}_{split}_wilc",
                path=os.path.join("tex", "por_modelo", model.replace(":", "_"))
            )

            # Friedman + Holm por contexto
            for c in ["com contexto", "sem contexto"]:
                sub = df[df.contexto == c]
                pivot = sub.pivot(index="fold", columns="shots", values="bert_f1").dropna()
                if pivot.shape[1] > 1 and pivot.shape[0] > 1:
                    stat, pvalue = friedmanchisquare(*[pivot[col] for col in pivot.columns])
                    tab = pd.DataFrame({
                        "Contexto": [c],
                        "Estatística": [stat],
                        "p-valor": [pvalue],
                        "Significativo": ["Sim" if pvalue < 0.05 else "Não"]
                    })
                    salvar(
                        tab,
                        f"{model}_{corpus}_{split}_friedman_{c.replace(' ', '_')}",
                        f"Friedman para {model} / {corpus}/{split} / {c}",
                        f"tab:{model}_{corpus}_{split}_fr_{c.replace(' ', '_')}",
                        path=os.path.join("tex", "por_modelo", model.replace(":", "_"))
                    )

                    if pvalue < 0.05:
                        shots = pivot.columns.tolist()
                        N = pivot.shape[0]
                        k = len(shots)

                        ranks = pivot.rank(axis=1, method="average", ascending=False)
                        mean_ranks = ranks.mean(axis=0)

                        pares = list(combinations(shots, 2))
                        z_vals = []
                        p_vals = []
                        comparacoes = []

                        SE = np.sqrt(k * (k + 1) / (6 * N))

                        for s1, s2 in pares:
                            Ri, Rj = mean_ranks[s1], mean_ranks[s2]
                            z = abs(Ri - Rj) / SE
                            p = 2 * (1 - norm.cdf(z))
                            z_vals.append(z)
                            p_vals.append(p)
                            comparacoes.append((s1, s2))

                        reject, p_corr, _, _ = multipletests(p_vals, alpha=0.05, method='holm')

                        df_holm = pd.DataFrame({
                            "Shots 1": [a for a, _ in comparacoes],
                            "Shots 2": [b for _, b in comparacoes],
                            "z": z_vals,
                            "p-valor bruto": p_vals,
                            "p-valor corrigido (Holm)": p_corr,
                            "Rejeita H0": reject
                        })

                        salvar(
                            df_holm,
                            f"{model}_{corpus}_{split}_holm_{c.replace(' ', '_')}",
                            f"Holm para {model} / {corpus}/{split} / {c}",
                            f"tab:{model}_{corpus}_{split}_holm_{c.replace(' ', '_')}",
                            path=os.path.join("tex", "por_modelo", model.replace(":", "_"))
                        )


# Parte 2: comparação entre modelos

def carregar_dados_modelos(base, prompts, corpus, split, models):
    registros = []
    for model in models:
        for fold in range(5):
            fold_path = os.path.join(base, model, corpus, split, f"fold_{fold}")
            if not os.path.isdir(fold_path):
                continue
            for p in prompts:
                fpath = os.path.join(fold_path, p + ".json")
                if not os.path.isfile(fpath):
                    continue
                with open(fpath, encoding="utf-8") as f:
                    j = json.load(f)

                # Obtém os valores médios das métricas (em %)
                bert_f1 = np.mean(j["scores"]["bertscore"]["f1"]) * 100
                bert_precision = np.mean(j["scores"]["bertscore"]["precision"]) * 100
                bert_recall = np.mean(j["scores"]["bertscore"]["recall"]) * 100
                bleu = j["scores"]["bleu"]["bleu"] * 100
                meteor = j["scores"]["meteor"]["meteor"] * 100

                registros.append({
                    "modelo": model,
                    "prompt": p,
                    "shots": shots_of(p),
                    "contexto": contexto_of(p),
                    "fold": fold,
                    "bert_f1": bert_f1,
                    "bert_precision": bert_precision,
                    "bert_recall": bert_recall,
                    "bleu": bleu,
                    "meteor": meteor,
                    "corpus": corpus,
                    "split": split
                })
    return pd.DataFrame(registros)


def salvar_formatado(df, fname, caption, label, corpus, split):
    import numpy as np

    path = os.path.join("tex", "comparacao_modelos", corpus, split)
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, fname + ".tex")

    # Detecta todas as colunas de métricas no formato "X (avg ± std)"
    metric_cols = [col for col in df.columns if "avg ± std" in col]

    # Extrai média de cada métrica para facilitar comparação
    mean_cols = {}
    for col in metric_cols:
        mean_cols[col] = df[col].apply(lambda x: float(x.split(" ± ")[0]))

    # Índices de destaque (negrito) para cada métrica e shots
    negrito_idx = {col: set() for col in metric_cols}
    for col in metric_cols:
        # Melhor para shots == 0
        subset_0 = df[df["Shots"] == 0]
        if not subset_0.empty:
            max_val_0 = mean_cols[col][subset_0.index].max()
            negrito_idx[col].update(subset_0[mean_cols[col][subset_0.index] == max_val_0].index)
        # Melhor para shots > 0
        subset_gt0 = df[df["Shots"] > 0]
        if not subset_gt0.empty:
            max_val_gt0 = mean_cols[col][subset_gt0.index].max()
            negrito_idx[col].update(subset_gt0[mean_cols[col][subset_gt0.index] == max_val_gt0].index)

    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{" + caption + "}\n")
        f.write("\\label{" + label + "}\n")

        col_format = "llc" + "c" * len(metric_cols)
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\hline\n")

        header = ["Modelo", "Contexto", "Shots"] + [m.split(" (")[0] + " (\\%)" for m in metric_cols]
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\hline\n")

        for model in df["Model"].unique():
            df_model = df[df["Model"] == model]
            n_model_rows = df_model.shape[0]

            # Para sublinhado: melhores por métrica dentro do modelo, excluindo negrito global
            sublinhado_idx = {}
            for col in metric_cols:
                subset = df_model.index.difference(negrito_idx[col])
                if len(subset) > 0:
                    max_val = mean_cols[col][subset].max()
                    sublinhado_idx[col] = set(df_model[mean_cols[col][df_model.index] == max_val].index).intersection(subset)
                else:
                    sublinhado_idx[col] = set()

            first_model_row = True

            for contexto in ["com contexto", "sem contexto"]:
                df_contexto = df_model[df_model["Context"] == contexto]
                if df_contexto.empty:
                    continue

                n_context_rows = df_contexto.shape[0]
                first_context_row = True

                for idx, row in df_contexto.iterrows():
                    line = ""

                    if first_model_row:
                        line += f"\\multirow{{{n_model_rows}}}{{*}}{{{model}}} & "
                        first_model_row = False
                    else:
                        line += " & "

                    if first_context_row:
                        line += f"\\multirow{{{n_context_rows}}}{{*}}{{{contexto}}} & "
                        first_context_row = False
                    else:
                        line += " & "

                    line += f"{int(row['Shots'])}"

                    # Aplica formatação individualmente por métrica
                    for m in metric_cols:
                        val = row[m]
                        if idx in negrito_idx[m]:
                            val = f"\\textbf{{{val}}}"
                        elif idx in sublinhado_idx[m]:
                            val = f"\\underline{{{val}}}"
                        line += f" & {val}"

                    line += " \\\\\n"
                    f.write(line)

                if contexto == "com contexto" and "sem contexto" in df_model["Context"].values:
                    f.write("\\cline{2-" + str(3 + len(metric_cols)) + "}\n")

            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")





def analisar_comparacao_modelos(df, corpus, split):
    modelos = df["modelo"].unique()

    # Resumo por modelo/contexto/shots
    # Resumo por modelo/contexto/shots com formatação "avg ± std"
    resumo = df.groupby(["modelo", "contexto", "shots"])["bert_f1"].agg(["mean", "std"]).reset_index()
    resumo["F1 (avg ± std)"] = resumo.apply(lambda x: f"{x['mean']:.2f} ± {x['std']:.2f}", axis=1)
    resumo = resumo[["modelo", "contexto", "shots", "F1 (avg ± std)"]]
    resumo.columns = ["Model", "Context", "Shots", "F1 (avg ± std)"]

    # Salvar em formato com multirow para model e context
    salvar_formatado(
        resumo,
        f"resumo_modelos_{corpus}_{split}_formatado",
        f"Summary of BERTScore (F1) by Model, Context, and Number of Shots ({corpus}/{split})",
        f"tab:resumo_formatado_{corpus}_{split}",
        corpus, split
    )


    # Friedman por contexto e shots, só se tiver 3 ou mais modelos
    friedman_rows = []

    for contexto in df["contexto"].unique():
        for shots in sorted(df["shots"].unique()):
            df_sub = df[(df["contexto"] == contexto) & (df["shots"] == shots)]
            pivot = df_sub.pivot(index="fold", columns="modelo", values="bert_f1").dropna()

            if pivot.shape[1] >= 3 and pivot.shape[0] > 1:
                stat, pvalue = friedmanchisquare(*[pivot[m] for m in pivot.columns])
                friedman_rows.append({
                    "Contexto": contexto,
                    "Shots": shots,
                    "Estatística": stat,
                    "p-valor": pvalue,
                    "Significativo": "Sim" if pvalue < 0.05 else "Não"
                })

                if pvalue < 0.05:
                    modelos = pivot.columns.tolist()
                    N = pivot.shape[0]  # número de folds
                    k = pivot.shape[1]  # número de modelos

                    # Calcula ranks por fold
                    ranks = pivot.rank(axis=1, method="average", ascending=False)  # rank alto = melhor
                    mean_ranks = ranks.mean(axis=0)

                    # Comparações entre pares
                    pares = list(combinations(modelos, 2))
                    z_vals = []
                    p_vals = []
                    comparacoes = []

                    SE = np.sqrt(k * (k + 1) / (6 * N))  # erro padrão da fórmula de Demšar

                    for m1, m2 in pares:
                        Ri, Rj = mean_ranks[m1], mean_ranks[m2]
                        z = abs(Ri - Rj) / SE
                        p = 2 * (1 - norm.cdf(z))  # teste bicaudal
                        z_vals.append(z)
                        p_vals.append(p)
                        comparacoes.append((m1, m2))

                    # Correção Holm
                    reject, p_corr, _, _ = multipletests(p_vals, alpha=0.05, method='holm')

                    df_holm = pd.DataFrame({
                        "Modelo 1": [m1 for m1, _ in comparacoes],
                        "Modelo 2": [m2 for _, m2 in comparacoes],
                        "z": z_vals,
                        "p-valor bruto": p_vals,
                        "p-valor corrigido (Holm)": p_corr,
                        "Rejeita H0": reject
                    })

                    salvar_comp(
                        df_holm,
                        f"holm_modelos_{corpus}_{split}_{contexto.replace(' ', '_')}_{shots}shots",
                        f"Pós-teste Friedman (Holm) para modelos - {contexto} - {shots} shots - {corpus}/{split}",
                        f"tab:holm_modelos_{corpus}_{split}_{contexto.replace(' ', '_')}_{shots}",
                        corpus, split
                    )

    df_friedman = pd.DataFrame(friedman_rows)
    salvar_comp(
        df_friedman,
        f"friedman_modelos_{corpus}_{split}",
        f"Teste Friedman entre modelos por contexto e shots para {corpus}/{split}",
        f"tab:friedman_modelos_{corpus}_{split}",
        corpus, split
    )

    # --- NOVO: Comparação entre modelos considerando TODOS os shots juntos, separado por contexto ---
    for contexto in df["contexto"].unique():
        # Filtra apenas pelo contexto, todos os shots juntos (ou seja, sem filtrar por shots)
        df_sub = df[df["contexto"] == contexto]

        # Pivotar para ter colunas por modelo e linhas por fold e shots combinados
        # Para evitar duplicatas, vamos usar fold e shots como índice juntos
        # Isso cria índice multi-nível para que não haja duplicatas para pivot
        df_sub_pivot = df_sub.pivot_table(
            index=["fold", "shots"],  # índice duplo para evitar duplicatas
            columns="modelo",
            values="bert_f1",
            aggfunc='mean'  # caso haja duplicatas, tira a média
        ).dropna()

        if df_sub_pivot.shape[1] >= 3 and df_sub_pivot.shape[0] > 1:
            stat, pvalue = friedmanchisquare(*[df_sub_pivot[m] for m in df_sub_pivot.columns])
            signif = "Sim" if pvalue < 0.05 else "Não"

            # Salvar resultado Friedman geral (todos os shots juntos)
            df_friedman_geral = pd.DataFrame([{
                "Contexto": contexto,
                "Estatística": stat,
                "p-valor": pvalue,
                "Significativo": signif
            }])
            salvar_comp(
                df_friedman_geral,
                f"friedman_modelos_geral_{corpus}_{split}_{contexto.replace(' ', '_')}",
                f"Teste Friedman entre modelos (todos os shots juntos) - {contexto} - {corpus}/{split}",
                f"tab:friedman_modelos_geral_{corpus}_{split}_{contexto.replace(' ', '_')}",
                corpus, split
            )

            if pvalue < 0.05:
                modelos = df_sub_pivot.columns.tolist()
                N = df_sub_pivot.shape[0]  # número de "fold+shots"
                k = len(modelos)

                # Ranks por linha (fold+shots)
                ranks = df_sub_pivot.rank(axis=1, method="average", ascending=False)
                mean_ranks = ranks.mean(axis=0)

                pares = list(combinations(modelos, 2))
                z_vals = []
                p_vals = []
                comparacoes = []
                SE = np.sqrt(k * (k + 1) / (6 * N))

                for m1, m2 in pares:
                    Ri, Rj = mean_ranks[m1], mean_ranks[m2]
                    z = abs(Ri - Rj) / SE
                    p = 2 * (1 - norm.cdf(z))
                    z_vals.append(z)
                    p_vals.append(p)
                    comparacoes.append((m1, m2))

                reject, p_corr, _, _ = multipletests(p_vals, alpha=0.05, method='holm')

                df_holm_geral = pd.DataFrame({
                    "Modelo 1": [m1 for m1, _ in comparacoes],
                    "Modelo 2": [m2 for _, m2 in comparacoes],
                    "z": z_vals,
                    "p-valor bruto": p_vals,
                    "p-valor corrigido (Holm)": p_corr,
                    "Rejeita H0": reject
                })

                salvar_comp(
                    df_holm_geral,
                    f"holm_modelos_geral_{corpus}_{split}_{contexto.replace(' ', '_')}",
                    f"Pós-teste Friedman (Holm) para modelos (todos shots juntos) - {contexto} - {corpus}/{split}",
                    f"tab:holm_modelos_geral_{corpus}_{split}_{contexto.replace(' ', '_')}",
                    corpus, split
                )


    # Comparação entre melhor zero-shot e melhor com mais shots
    df_avg = df.groupby(["modelo", "contexto", "shots"])["bert_f1"].mean().reset_index()

    df_avg_0 = df_avg[df_avg["shots"] == 0]
    df_avg_gt0 = df_avg[df_avg["shots"] > 0]

    if not df_avg_0.empty and not df_avg_gt0.empty:
        best_0 = df_avg_0.loc[df_avg_0["bert_f1"].idxmax()]
        best_gt0 = df_avg_gt0.loc[df_avg_gt0["bert_f1"].idxmax()]

        modelo_0 = best_0["modelo"]
        contexto_0 = best_0["contexto"]
        shots_0 = int(best_0["shots"])

        modelo_gt0 = best_gt0["modelo"]
        contexto_gt0 = best_gt0["contexto"]
        shots_gt0 = int(best_gt0["shots"])

        dados_0 = df[
            (df["modelo"] == modelo_0) &
            (df["contexto"] == contexto_0) &
            (df["shots"] == shots_0)
        ].sort_values("fold")["bert_f1"].values

        dados_gt0 = df[
            (df["modelo"] == modelo_gt0) &
            (df["contexto"] == contexto_gt0) &
            (df["shots"] == shots_gt0)
        ].sort_values("fold")["bert_f1"].values

        if len(dados_0) == len(dados_gt0) and len(dados_0) > 1:
            stat, pvalue = wilcoxon(dados_0, dados_gt0)
            significativo = "Sim" if pvalue < 0.05 else "Não"
        else:
            stat, pvalue, significativo = np.nan, np.nan, "N/A"

        df_comp_best = pd.DataFrame([{
            "Modelo zero-shot": modelo_0,
            "Contexto zero-shot": contexto_0,
            "Shots zero-shot": shots_0,
            "Modelo mais-shots": modelo_gt0,
            "Contexto mais-shots": contexto_gt0,
            "Shots mais-shots": shots_gt0,
            "Estatística Wilcoxon": stat,
            "p-valor": pvalue,
            "Significativo?": significativo
        }])

        salvar_comp(
            df_comp_best,
            f"melhor_0_vs_mais_shots_{corpus}_{split}",
            f"Wilcoxon entre melhor zero-shot e melhor modelo com mais shots ({corpus}/{split})",
            f"tab:melhor_0_vs_mais_shots_{corpus}_{split}",
            corpus, split
        )

def analisar_comparacao_modelos_todas_metricas(df, corpus, split):
    df = df.copy()

    # Agrupa e calcula média e desvio padrão de cada métrica
    resumo = df.groupby(["modelo", "contexto", "shots"]).agg({
        "bert_f1": ["mean", "std"],
        "bert_precision": ["mean", "std"],
        "bert_recall": ["mean", "std"],
        "bleu": ["mean", "std"],
        "meteor": ["mean", "std"]
    }).reset_index()

    # Formata colunas com "avg ± std"
    resumo["Precision (avg ± std)"] = resumo.apply(lambda x: f"{x[('bert_precision', 'mean')]:.2f} ± {x[('bert_precision', 'std')]:.2f}", axis=1)
    resumo["Recall (avg ± std)"] = resumo.apply(lambda x: f"{x[('bert_recall', 'mean')]:.2f} ± {x[('bert_recall', 'std')]:.2f}", axis=1)
    resumo["F1 (avg ± std)"] = resumo.apply(lambda x: f"{x[('bert_f1', 'mean')]:.2f} ± {x[('bert_f1', 'std')]:.2f}", axis=1)   
    resumo["BLEU (avg ± std)"] = resumo.apply(lambda x: f"{x[('bleu', 'mean')]:.2f} ± {x[('bleu', 'std')]:.2f}", axis=1)
    resumo["METEOR (avg ± std)"] = resumo.apply(lambda x: f"{x[('meteor', 'mean')]:.2f} ± {x[('meteor', 'std')]:.2f}", axis=1)

    # Seleciona colunas finais e renomeia
    resumo = resumo[["modelo", "contexto", "shots",
                     "Precision (avg ± std)", 
                     "Recall (avg ± std)", "F1 (avg ± std)",  "BLEU (avg ± std)", "METEOR (avg ± std)"]]
    
    resumo.columns = ["Model", "Context", "Shots",
                      "Precision (avg ± std)",
                      "Recall (avg ± std)", "F1 (avg ± std)",  "BLEU (avg ± std)", "METEOR (avg ± std)"]

    salvar_formatado(
        resumo,
        f"resumo_modelos_{corpus}_{split}_completo",
        f"Summary of BERTScore + BLEU + METEOR by Model, Context, and Number of Shots ({corpus}/{split})",
        f"tab:resumo_completo_{corpus}_{split}",
        corpus, split
    )



for corpus in corpora:
    for split in splits:
        df_comparacao = carregar_dados_modelos(
            base=base,
            prompts=prompts,
            corpus=corpus,
            split=split,
            models=models
        )
        if not df_comparacao.empty:
            analisar_comparacao_modelos(df_comparacao, corpus, split)
            analisar_comparacao_modelos_todas_metricas(df_comparacao, corpus, split)
