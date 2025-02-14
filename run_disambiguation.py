import os
import time
import json
import numpy as np
import pandas as pd
from openai import OpenAI
from zhipuai import ZhipuAI
from generate_qs import QuestionGenerator
from logger import ExperimentLogger
from llm import BinaryJudgeGenerator, BinaryJudgeEvaluator,\
                MatchRateGenerator, MatchRateEvaluator \
                


def run_disambiguation(mycls, data_scale, genName, genClient, evalName, evalClient):
    input_dir = f"./output/question/{mycls}/"
    output_dir = f"./output/{mycls}/{genName}/"
    os.makedirs(output_dir, exist_ok=True)

    #######################################
    # 初始化Logger
    experiment_logger = ExperimentLogger(log_dir="./logs", log_file_prefix="my_experiment")

    # 记录实验参数
    experiment_params = {
        "Type": "DisAmbiguation",
        "Generation Model": genName,
        "Evaluation Model": evalName,
        "Class": mycls,
        "Input Directory": input_dir,
        "Output Directory": output_dir,
    }
    experiment_logger.log_experiment_params(experiment_params)
    #######################################

    # 直接读取问题
    with open(f"{input_dir}binary_judge.json", "r", encoding="utf-8") as f:
        disamb2qa = json.load(f)

    biGenerator = BinaryJudgeGenerator(genClient, genName) #回答问题
    biEvaluator = BinaryJudgeEvaluator(evalClient, evalName) #判断答案是否正确

    questions, answers = [], []
    for disambEntityUrl, info in disamb2qa.items():
        questions.extend([info["questions"][0][0], info["questions"][1][0]])
        answers.extend([info["questions"][0][1], info["questions"][1][1]])
    predictions = biGenerator.generate(questions)
    pos_precision, neg_precision, cross_precision, all_precision = biEvaluator.eval(answers, predictions)

    ### output
    print(f"########True Postive: {pos_precision}########")
    print(f"########True Negative: {neg_precision}########")
    print(f"########Pair Accuracy: {cross_precision}########")
    print(f"########Accuracy: {all_precision}########")

    data = {"question": questions, "prediction": predictions, "answer": answers}
    df = pd.DataFrame(data)
    output_file = f"{output_dir}BinaryJudge.csv"
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', index=False, header=False, encoding="utf-8")
    else:
        df.to_csv(output_file, index=False, encoding="utf-8")

    experiment_logger.log_info("Running Binary Judge...")
    experiment_logger.log_results("Binary Judge Evaluation", {
        "Questions Generated": len(questions),
        "True Postive": pos_precision,
        "True Negative": neg_precision,
        "Pair Accuracy": cross_precision,
        "Accuracy": all_precision,
    })
    experiment_logger.log_info(f"Results saved to: {output_dir}BinaryJudge.csv")

    matchGenerator = MatchRateGenerator(genClient, genName)
    matchEvaluator = MatchRateEvaluator(evalClient, evalName)
    # 直接读取问题
    with open(f"{input_dir}match_rate.json", "r", encoding="utf-8") as f:
        match_disamb2qa = json.load(f)

    questions, answers = [], []
    for disambEntityUrl, qa in match_disamb2qa.items():
        questions.append(qa[0]); answers.append(qa[1])
    predictions = matchGenerator.generate(questions)
    MR = matchEvaluator.eval(answers, predictions)

    ### output
    print(f"MR: {MR}")

    data = {"question": questions, "prediction": predictions, "answer": answers}
    df = pd.DataFrame(data)
    output_file = f"{output_dir}MatchRate.csv"
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', index=False, header=False, encoding="utf-8")
    else:
        df.to_csv(output_file, index=False, encoding="utf-8")

    # 记录Unique Answer Evaluation结果
    experiment_logger.log_info("Running Match Rate...")
    experiment_logger.log_results("Match Rate Evaluation", {
        "Questions Generated": len(questions),
        "MR": MR,
    })
    experiment_logger.log_info(f"Results saved to: {output_dir}MatchRate.csv")

    # 实验完成
    experiment_logger.log_info("Disambiguation Experiment Completed")


if __name__ == "__main__":
    cls_list = ["Person", "Place", "Organisation", "Building", "Work", "MultiClass"]
    data_scale = 100

    ## Testing Model
    genName = ""
    genClient = OpenAI(
        api_key="",
        base_url=""
    )


    for cls in cls_list:
        run_disambiguation(cls, data_scale, genName, genClient, genName, genClient)
        time.sleep(5)