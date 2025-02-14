import os
import numpy as np
import pandas as pd
from openai import OpenAI
from zhipuai import ZhipuAI
from generate_qs import QuestionGenerator
from llm import (BaseGenerator, MultiAnswerEvaluator,
                UniqueAnswerEvaluator, EntityExistEvaluator, 
                BinaryJudgeGenerator, BinaryJudgeEvaluator, 
                MatchRateGenerator, MatchRateEvaluator, 
                InfoCompleteGenerator, InfoCompleteEvaluator, 
                MultiTurnGenerator)
                

########## 在这里调换测试的模型 ##########
genClient = OpenAI(
     api_key="sk-e7bbc4456cbc445abfb7b024b6126e2d",
     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
genName = "llama3.3-70b-instruct"
#######################################

evalClient = ZhipuAI(api_key="c0639fda4d24eb4ce124cd890babeb81.tu3IZ4xOuwxd3J4s")
evalName = "glm-4-flash"

########## 设置参数 ####################
mycls = "Person"
data_scale = 20
output_dir = f"./output/{mycls}/{genName}/"
#######################################

question_generator = QuestionGenerator(mycls, data_scale)
generator = BaseGenerator(genClient, genName) #回答问题
unique_evaluator = UniqueAnswerEvaluator(evalClient, evalName)
uniqueQA = question_generator.genq_uniq_answer()
uniPredictions = generator.generate(list(uniqueQA.values()))
uniPrecision, uniOutputList = unique_evaluator.eval(list(uniqueQA.keys()), uniPredictions)
print(f"########Unique Precision: {uniPrecision}########")

data = {"name": list(uniqueQA.keys()), "question": list(uniqueQA.values()), \
        "prediction": uniPredictions, "correct/wrong": uniOutputList}
df = pd.DataFrame(data)
output_file = f"{output_dir}UniqueAnswer.csv"
# 检查文件是否已存在
if os.path.exists(output_file):
    # 如果文件存在，追加模式写入，不写表头
    df.to_csv(output_file, mode='a', index=False, header=False, encoding="utf-8")
else:
    # 如果文件不存在，写入并包含表头
    df.to_csv(output_file, index=False, encoding="utf-8")


multi_evaluator = MultiAnswerEvaluator(evalClient, evalName)
multiQA, links = question_generator.genq_multi_answer()
multiPredictions = generator.generate(list(multiQA.values()))
multiPrecision, multiOutputList = multi_evaluator.eval(list(multiQA.keys()), multiPredictions)
print(f"########Multi Precision: {multiPrecision}########")

print(f"########ADR: {multiPrecision-1+uniPrecision}########")

data = {"name": list(multiQA.keys()), "question": list(multiQA.values()), \
        "prediction": multiPredictions, "correct/wrong": multiOutputList}
df = pd.DataFrame(data)
output_file = f"{output_dir}MultiAnswer.csv"
# 检查文件是否已存在
if os.path.exists(output_file):
    # 如果文件存在，追加模式写入，不写表头
    df.to_csv(output_file, mode='a', index=False, header=False, encoding="utf-8")
else:
    # 如果文件不存在，写入并包含表头
    df.to_csv(output_file, index=False, encoding="utf-8")

recallEvaluator = EntityExistEvaluator(evalClient, evalName)
ARR, outputList = recallEvaluator.eval(links, multiPredictions)
print(f"########ARR: {ARR}########")
data = {"name": links, "prediction": multiPredictions, "hits": outputList} 
df = pd.DataFrame(data)
output_file = f"{output_dir}Recall.csv"
if os.path.exists(output_file):
    # 如果文件存在，追加模式写入，不写表头
    df.to_csv(output_file, mode='a', index=False, header=False, encoding="utf-8")
else:
    # 如果文件不存在，写入并包含表头
    df.to_csv(output_file, index=False, encoding="utf-8")
    