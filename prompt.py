import sys
sys.stderr = sys.stdout

import os
import json
import pandas as pd

from evaluate import load
from models.DecoderResponse import DecoderResponse
from tqdm import tqdm

# Inicializa o modelo
#model_name = "gemma3:27b"
#model = DecoderResponse(model_name)

# Carrega métricas
bertscore = load("bertscore")
bleu = load("bleu")
meteor = load("meteor")


# Lista completa de prompts
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
    "context_8_shot",]
# Nome do corpus
#models = ["qwen3:30b", "falcon:40b", "mixtral:8x7b"]

models = ["gaia:4b", "mistral:7b", "phi3:14b", "llama3.1:8b", "gemma3:12b", "qwen3:14b"]
corpora = ["definitions_bete", "definitions_geocorpus"]
splits = ["iid"]

for model_name in models:
    model = DecoderResponse(model_name)
    for corpus in corpora:
        for split_type in splits:
            corpus_path = f"{corpus}/{split_type}"
            # Private local outputs: contain reference definitions and model text.
            results_base_path = f"local_raw_results/{model_name.replace(':', '-')}/{corpus}/{split_type}"

            # Loop pelos folds
            for fold_num in range(5):
                test_path = os.path.join(corpus_path, f"fold_{fold_num}", "test.json")
                
                if not os.path.exists(test_path):
                    print(f"[AVISO] Arquivo não encontrado: {test_path}")
                    continue

                df = pd.read_json(test_path)
                if "scope" not in df.columns:
                    print(f"[ERRO] Coluna 'scope' ausente no arquivo {test_path}")
                    continue

                true_definitions = df["scope"].tolist()

                for prompt_name in prompt_names:
                    if prompt_name not in df.columns:
                        print(f"[AVISO] Prompt '{prompt_name}' não encontrado em {test_path}")
                        continue

                    pred_definitions = []

                    print(f"[INFO] Processando {corpus} {split_type} | fold {fold_num} | prompt: {prompt_name} | total: {len(df)}")

                    #for i, row in tqdm(df.iterrows()):
                    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
                        prompt = row[prompt_name]
                        #print("Prompt:", prompt)
                        output = model.train(prompt)
                        pred_definitions.append(output.strip())
                        #print("=======")

                    # Avaliação
                    bert_results = bertscore.compute(
                        predictions=pred_definitions,
                        references=true_definitions,
                        model_type="bert-base-multilingual-cased",
                        lang="pt"
                    )

                    # Cálculo das médias de precision, recall e f1
                    avg_precision = sum(bert_results["precision"]) / len(bert_results["precision"])
                    avg_recall = sum(bert_results["recall"]) / len(bert_results["recall"])
                    avg_f1 = sum(bert_results["f1"]) / len(bert_results["f1"])

                    # Resultados do BLEU e METEOR
                    bleu_results = bleu.compute(
                        predictions=pred_definitions,
                        references=[[ref] for ref in true_definitions]
                    )
                    meteor_results = meteor.compute(
                        predictions=pred_definitions,
                        references=true_definitions
                    )

                    result_data = {
                        "true_definitions": true_definitions,
                        "pred_definitions": pred_definitions,
                        "scores": {
                            "bertscore": {
                                "precision": bert_results["precision"],
                                "recall": bert_results["recall"],
                                "f1": bert_results["f1"],
                                "average": {
                                    "precision": avg_precision,
                                    "recall": avg_recall,
                                    "f1": avg_f1
                                }
                            },
                            "bleu": bleu_results,
                            "meteor": meteor_results
                        }
                    }

                    # Impressão dos resultados médios
                    print(f"[RESULTADOS MÉDIOS] Fold {fold_num} | Prompt: {prompt_name}")
                    print(f"  ➤ BERTScore - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
                    print(f"  ➤ BLEU: {bleu_results['bleu']:.4f}")
                    print(f"  ➤ METEOR: {meteor_results['meteor']:.4f}")

                    # Salvar resultados
                    output_dir = os.path.join(results_base_path, f"fold_{fold_num}")
                    os.makedirs(output_dir, exist_ok=True)

                    output_file = os.path.join(output_dir, f"{prompt_name}.json")
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=2)

                    print(f"[OK] Resultados salvos em: {output_file}")
