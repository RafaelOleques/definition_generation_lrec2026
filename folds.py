from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def criar_prompt_zero_shot(contexto, entidade, type_terms="healthcare", expert="health", source="online healthcare forums on diabetes"):
    return (
        f"Act as a {expert} expert drafting official definitions for {type_terms} terms found in {source}.\n"
        f"Define the term: '{entidade}', based on the text provided between the triple dashes. "
        "\n---\n"
        f"{contexto}\n"
        "---\n"
        "Provide a clear and concise definition that accurately conveys the "
        "meaning of the term within the context of the sentence.\n"
        "The definition must be written in Portuguese.\n"
        "ONLY return the definition text. Do not include any interpretations, comments, or additional formatting."
    )

def criar_prompt_context_example(contexto, entidade, examples, type_terms="healthcare", expert="health", source="online healthcare forums on diabetes"):
    return (
        f"Act as a {expert} expert drafting official definitions for {type_terms} terms found in {source}.\n"
        f"Define the term: '{entidade}', based on the text provided between the triple dashes."
        "\n---\n"
        f"{contexto}\n"
        "---\n"
        "Here are some examples between the triple dashes."
        "\n---\n"
        f"{examples}\n"
        "---\n"
        "Provide a clear and concise definition that accurately conveys the "
        "meaning of the term within the context of the sentence.\n"
        "The definition must be written in Portuguese.\n"
        "ONLY return the definition text. Do not include any interpretations, comments, or additional formatting."
    )

def criar_prompt_no_context_example(entidade, examples, type_terms="healthcare", expert="health", source="online healthcare forums on diabetes"):
    return (
        f"Act as a {expert} expert drafting official definitions for {type_terms} terms found in {source}.\n"
        f"Define the term: '{entidade}'."
        "\nHere are some examples between the triple dashes."
        "\n---\n"
        f"{examples}\n"
        "---\n"
        "Provide a clear and concise definition that accurately conveys the "
        "meaning of the term within the context of the sentence.\n"
        "The definition must be written in Portuguese.\n"
        "ONLY return the definition text. Do not include any interpretations, comments, or additional formatting."
    )

def criar_prompt_no_context_no_example(entidade, type_terms="healthcare", expert="health", source="online healthcare forums on diabetes"):
    return (
        f"Act as a {expert} expert drafting official definitions for {type_terms} terms found in {source}.\n"
        f"Define the term: '{entidade}'."
        "Provide a clear and concise definition that accurately conveys the "
        "meaning of the term within the context of the sentence.\n"
        "The definition must be written in Portuguese.\n"
        "ONLY return the definition text. Do not include any interpretations, comments, or additional formatting."
    )

def build_term_only_retrieval(df_train, df_test,
                              type_terms="healthcare", expert="health", source="online healthcare forums on diabetes", model_name='Qwen/Qwen3-Embedding-4B',):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    import numpy as np

    model = SentenceTransformer(model_name)

    unique_train = df_train[df_train['term'].duplicated(keep=False) == False].reset_index(drop=True)
    train_terms = unique_train['term'].tolist()
    test_terms = df_test['term'].tolist()

    train_embeddings = model.encode(train_terms, convert_to_tensor=True)
    test_embeddings = model.encode(test_terms, convert_to_tensor=True)

    retrievals = []

    for idx, test_term in tqdm(enumerate(test_terms), total=len(test_terms), desc="Building prompts"):
        query_embedding = test_embeddings[idx].unsqueeze(0)
        similarities = cosine_similarity(query_embedding.cpu().numpy(), train_embeddings.cpu().numpy())[0]

        top_indices = np.argsort(similarities)[::-1]
        top_indices = [i for i in top_indices if unique_train.iloc[i]['term'] != test_term]

        test_row = df_test.iloc[idx]
        contexto = ""
        val = test_row.get("sentences_containing_term", "")
        if isinstance(val, list) and val:
            contexto = val[0]
        elif isinstance(val, (str, np.ndarray)):
            contexto = val[0] if isinstance(val, np.ndarray) and len(val) else str(val)

        # ➤ Exemplo formatado com contexto
        exemplos_com_contexto = []
        # ➤ Exemplo formatado sem contexto
        exemplos_sem_contexto = []

        for i in top_indices:
            row = unique_train.iloc[i]
            escopo = row.get("scope", "")
            val = row.get("sentences_containing_term", "")
            sent = ""
            if isinstance(val, list) and val:
                sent = val[0]
            elif isinstance(val, (str, np.ndarray)):
                sent = val[0] if isinstance(val, np.ndarray) and len(val) else str(val)

            exemplos_com_contexto.append(f"{row['term']}\nDefinition: {escopo.strip()}\nExample: {sent.strip()}")
            exemplos_sem_contexto.append(f"{row['term']}\nDefinition: {escopo.strip()}")

        prompt_dict = {
            "term": test_term,
            "context": contexto,
            "scope": test_row.get("scope", ""),
            "zero_shot": criar_prompt_zero_shot(contexto, test_term, type_terms, expert, source),
            "no_context_no_example": criar_prompt_no_context_no_example(test_term, type_terms, expert, source),
        }

        # ➤ No context com múltiplos exemplos
        for k in [1, 2, 4, 8]:
            if len(exemplos_sem_contexto) >= k:
                exemplos_k = "\n\n".join(exemplos_sem_contexto[:k])
                prompt_dict[f"no_context_{k}_shot"] = criar_prompt_no_context_example(
                    test_term, exemplos_k, type_terms, expert, source
                )

        # ➤ Com contexto e múltiplos exemplos
        for k in [1, 2, 4, 8]:
            if len(exemplos_com_contexto) >= k:
                exemplos_k = "\n\n".join(exemplos_com_contexto[:k])
                prompt_dict[f"context_{k}_shot"] = criar_prompt_context_example(
                    contexto, test_term, exemplos_k, type_terms, expert, source
                )

        retrievals.append(prompt_dict)

    return pd.DataFrame(retrievals)

def create_all_fold_dfs(definitions_bete, iid_folds,
                        type_terms="healthcare", expert="health",
                        source="online healthcare forums on diabetes"):
    iid_fold_dfs = []

    for fold_idx in range(5):
        print(f"\n[📁] Processando Fold {fold_idx}...")

        # IID fold
        print(f"  ➤ [IID] Preparando dados...")
        train_idx_iid, test_idx_iid = iid_folds[fold_idx]
        df_train_iid = definitions_bete.loc[train_idx_iid].reset_index(drop=True)
        df_test_iid = definitions_bete.loc[test_idx_iid].reset_index(drop=True)

        print(f"  ➤ [IID] Gerando prompts com build_term_only_retrieval()...")
        df_test_iid = build_term_only_retrieval(df_train_iid, df_test_iid, type_terms, expert, source)
        print(f"  ✅ [IID] Fold {fold_idx} pronto.")

        iid_fold_dfs.append({
            "fold": fold_idx,
            "type": "iid",
            "train": df_train_iid,
            "test": df_test_iid
        })

    return iid_fold_dfs


import json
from pathlib import Path

def load_folds_json(output_dir="folds_output"):
    output_dir = Path(output_dir)

    with open(output_dir / "iid_folds.json", "r", encoding="utf-8") as f:
        iid_json = json.load(f)

    return [(fold["train_indices"], fold["test_indices"]) for fold in iid_json]


