import os
import time
import json
import random
import numpy as np
import pandas as pd
from itertools import islice
from openai import OpenAI
from zhipuai import ZhipuAI
from generate_qs import QuestionGenerator
from logger import ExperimentLogger
from llm import (InfoCompleteGenerator, InfoCompleteEvaluator,
                MultiTurnGenerator, ContextualBot)

SEED = 42
random.seed(SEED)


def run_contextual(mycls, data_scale, genName, genClient, evalName, evalClient):
    input_dir = f"./output/question/{mycls}/"
    output_dir = f"./output/{mycls}/{genName}/"
    os.makedirs(output_dir, exist_ok=True)

    #######################################
    # 初始化Logger
    experiment_logger = ExperimentLogger(log_dir="./logs", log_file_prefix="my_experiment")

    # 记录实验参数
    experiment_params = {
        "Type": "Contextual Resolution",
        "Generation Model": genName,
        "Evaluation Model": evalName,
        "Class": mycls,
        "Input Directory": input_dir,
        "Output Directory": output_dir,
    }
    experiment_logger.log_experiment_params(experiment_params)
    #######################################

    with open(f"{input_dir}contextual_qa.json", "r", encoding='utf-8') as f:
        disamb2qa = json.load(f)

    # 取前n个元素
    disamb2qa = dict(islice(disamb2qa.items(), data_scale))

    questions, additionals, answers = [], [], []

    for disambEntityUrl in disamb2qa.keys():
        val_dicts = disamb2qa[disambEntityUrl]
        questions.append(val_dicts["qa"][0])
        answers.append(val_dicts["qa"][1])
        additionals.append(val_dicts["additionals"])

    contextualBot = ContextualBot(genClient, genName, evalClient, evalName)
    acc_before, acc_after, avg_turn, predictions, precisions = contextualBot.chat(questions, additionals, answers)
    data = {"question":questions, "answer":answers, "prediction":predictions, "precision": precisions}
    df = pd.DataFrame(data)
    output_file = f"{output_dir}ContextualResolution.csv"
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', index=False, header=False, encoding="utf-8")
    else:
        df.to_csv(output_file, index=False, encoding="utf-8")

    experiment_logger.log_info("Contextual Resolution...")
    experiment_logger.log_results("Contextual Resolution", {
        "Questions Generated": len(questions),
        "Accuracy Before": acc_before,
        "Accuracy After": acc_after,
        "Average Turns": avg_turn
    })
    experiment_logger.log_info(f"Results saved to: {output_dir}ContextualResolution.csv")
    experiment_logger.log_info("Contextual Resolution Experiment Completed")


if __name__ == "__main__":
    data_scale = 100
    cls_list = ["Person", "Place", "Organisation", "Building", "Work", "MultiClass"]

    ## Testing Model
    genName = ""
    genClient = OpenAI(
        api_key="",
        base_url=""
    )

    ## Evaluation Model
    evalName = ""
    evalClient = OpenAI(
        api_key="",
        base_url=""
    )

    for cls in cls_list:
        run_contextual(cls, data_scale, genName, genClient, evalName, evalClient)
        time.sleep(5)