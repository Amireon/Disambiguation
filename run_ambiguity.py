import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from zhipuai import ZhipuAI
from generate_qs import QuestionGenerator
from logger import ExperimentLogger
from llm import (BaseGenerator, MultiAnswerGenerator,
                MultiAnswerEvaluator, UniqueAnswerEvaluator, EntityExistEvaluator)               


def run_ambiguity(mycls, data_scale, genName, genClient, evalName, evalClient):
    output_dir = f"./output/{mycls}/{genName}/"
    os.makedirs(output_dir, exist_ok=True)

    #######################################
    # 初始化Logger
    experiment_logger = ExperimentLogger(log_dir="./logs", log_file_prefix="my_experiment")

    # 记录实验参数
    experiment_params = {
        "Type": "Ambiguity",
        "Generation Model": genName,
        "Evaluation Model": evalName,
        "Class": mycls,
        "Data Scale": data_scale,
        "Output Directory": output_dir,
    }
    experiment_logger.log_experiment_params(experiment_params)
    #######################################

    question_generator = QuestionGenerator(mycls, data_scale)
    multiQA, links = question_generator.genq_multi_answer()

    multiGenerator = BaseGenerator(genClient, genName)
    multiPredictions = multiGenerator.generate(list(multiQA.values()))
    multi_evaluator = MultiAnswerEvaluator(evalClient, evalName)
    multiPrecision, multiOutputList = multi_evaluator.eval(list(multiQA.keys()), multiPredictions)
    print(f"########ADR: {multiPrecision}########")

    data = {"name": list(multiQA.keys()), "question": list(multiQA.values()), \
            "prediction": multiPredictions, "correct/wrong": multiOutputList}
    df = pd.DataFrame(data)
    output_file = os.path.join(output_dir, "AmbiguityDiscover.csv")
    # 检查文件是否已存在
    if os.path.exists(output_file):
        # 如果文件存在，追加模式写入，不写表头
        df.to_csv(output_file, mode='a', index=False, header=False, encoding="utf-8")
    else:
        # 如果文件不存在，写入并包含表头
        df.to_csv(output_file, index=False, encoding="utf-8")

    # 记录Multi-Answer Evaluation结果
    experiment_logger.log_info("Running Ambiguity Discover Rate...")
    experiment_logger.log_results("Multi-Answer Evaluation", {
        "Questions Generated": len(multiQA),
        "ADR": multiPrecision,
    })
    experiment_logger.log_info(f"Results saved to: {output_dir}AmbiguityDiscover.csv")

    recallGenerator = MultiAnswerGenerator(genClient, genName)
    recallPredictions = recallGenerator.generate(list(multiQA.keys()))
    recallEvaluator = EntityExistEvaluator(evalClient, evalName)
    ARR, outputList = recallEvaluator.eval(links, recallPredictions)
    print(f"########ARR: {ARR}########")
    data = {"name": links, "prediction": recallPredictions, "hits": outputList}
    df = pd.DataFrame(data)
    output_file = os.path.join(output_dir, "AmbiguityRecall.csv")
    if os.path.exists(output_file):
        # 如果文件存在，追加模式写入，不写表头
        df.to_csv(output_file, mode='a', index=False, header=False, encoding="utf-8")
    else:
        # 如果文件不存在，写入并包含表头
        df.to_csv(output_file, index=False, encoding="utf-8")

    # 记录Unique Answer Evaluation结果
    experiment_logger.log_info("Running Ambiguity Recall Rate...")
    experiment_logger.log_results("Unique Answer Evaluation", {
        "Links Generated": len(links),
        "ARR": ARR,
    })
    experiment_logger.log_info(f"Results saved to: {output_dir}AmbiguityRecall.csv")

    # 实验完成
    experiment_logger.log_info("Ambiguity Experiment Completed")


if __name__ == "__main__":
    cls_list = ["Person", "Place", "Organisation", "Building", "Work", "MultiClass"]
    data_scale = 100

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

    for mycls in cls_list:
        run_ambiguity(mycls, data_scale, genName, genClient, evalName, evalClient)
        time.sleep(5)