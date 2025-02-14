import os
import re
import string
import numpy as np
from tqdm import tqdm


class BaseGenerator(object):
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def get_api_response(self, question:str):
        response = self.client.chat.completions.create(
            model = f"{self.model}",  # 填写需要调用的模型名称
            temperature = 0, 
            messages=[
                {"role": "system", 
                 "content": '''Based on stored knowledge, without performing external searches, answer the following question.
                               Notice: The model should keep responses as concise as possible, ideally under 200 tokens.'''},
                {"role": "user", "content": question}],
        )
        return response.choices[0].message.content

    def generate(self, questions):
        predictions = []
        for question in tqdm(questions, desc="Generating answers..."):
            prompt = question
            response = str(self.get_api_response(prompt))
            predictions.append(response)
            # time.sleep(6.5)
        return predictions

    def write_to_file(self, predictions, answers, output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            for idx in range(len(predictions)):
                pred = predictions[idx]
                ans = answers[idx]
                f.write(pred + "\t" + ans)


class MultiAnswerGenerator(BaseGenerator):
    def __init__(self, client, model):
        super().__init__(client, model)
    
    def get_api_response(self, name:str):
        response = self.client.chat.completions.create(
            temperature = 0, 
            model = f"{self.model}",  # 填写需要调用的模型名称
            messages=[
                {"role": "system", 
                 "content": ''' Based on your stored knowledge, without performing external searches, answer the following question.'''},
                {"role": "user", 
                 "content": f'''I will provide you with an ambiguous name. Please search your memory and list all the specific entities associated with this name. 
                                Make your response as complete as possible. For each entity, provide the following format:
                                    Entity Name : A concise description of the entity (no more than 50 words).
                                    Example:
                                        Ambiguous Name: “Apple”
                                            •	Apple Inc. : A multinational technology company that designs and sells consumer electronics and software.
                                            •	Apple (Fruit) : A sweet, edible fruit commonly grown in temperate regions.
                                    Notice: Ensure that the entity names are as complete and specific as possible to avoid ambiguity.
                                Now, here is the ambiguous name: {name}.'''},],
        )
        return response.choices[0].message.content


class BinaryJudgeGenerator(BaseGenerator):
    def __init__(self, client, model):
        super().__init__(client, model)
    
    def get_api_response(self, question:str):
        response = self.client.chat.completions.create(
            temperature = 0, 
            model = f"{self.model}",  # 填写需要调用的模型名称
            messages=[
                {"role": "system", 
                 "content": ''' Based on the given context and your stored knowledge, without performing external searches.
                                Notice: Output only "Yes" or "No", without any additional explanations.'''},
                {"role": "user", "content": question},],
        )
        return response.choices[0].message.content


class MatchRateGenerator(BaseGenerator):
    def __init__(self, client, model):
        super().__init__(client, model)
    
    def get_api_response(self, question:str):
        response = self.client.chat.completions.create(
            temperature = 0, 
            model = f"{self.model}",  # 填写需要调用的模型名称
            messages=[
                {"role": "system", 
                 "content": '''Based on the given context and your stored knowledge, without performing external searches, choose the correct option from the following choices (A, B, C, etc.).
                            Notice: 
                                1. Answer only with the option letter (A, B, C, etc.), without any additional explanations or text.
                                2. Notice: At the beginning of your response, **first provide the option letter** to make it easy to be identified.'''},## add
                {"role": "user", "content": question},],
        )
        return response.choices[0].message.content

class InfoCompleteGenerator(BaseGenerator):
    def __init__(self, client, model):
        super().__init__(client, model)
    
    def get_api_response(self, question:str):
        response = self.client.chat.completions.create(
            temperature = 0, 
            model = f"{self.model}",  # 填写需要调用的模型名称
            messages=[
                {"role": "system", 
                 "content": "Based on the given context and your stored knowledge, \
                             without performing external searches, answer the following question."},
                {"role": "user", "content": question},],
        )
        return response.choices[0].message.content


class MultiTurnGenerator(BaseGenerator):
    def __init__(self, client, model):
        super().__init__(client, model)
    
    def generate(self, questions, choices, num=4):
        '''
        params:
            num: int, the number of choices (max turns)
        '''
        def choose_from(response):
            '''
            func: 判断response是否是想从choices中选择一个选项
            '''
            # 如果没有选择任何一个选项，或者答案中写着：The answer is ...
            if response[0].lower() in string.ascii_lowercase[:num] or "answer" in response.lower():
                return True
            return False
        predictions, turns, complete_history = [], [], []
        labels = list(string.ascii_lowercase[:num])
        for i in tqdm(range(len(questions)), desc="Generating answers..."):
            history = []
            now_choice = choices[i]
            initial_prompt = questions[i]
            response = self.get_api_response(initial_prompt)
            history.append(f"User: {initial_prompt}")
            print(f"Initial question: {initial_prompt}") #print
            history.append(f"Model: {response}")
            print(f"First turn response: {response}") #print
            cnt = 0
            while True:
                if not choose_from(response):
                    predictions.append(response)
                    break
                cnt += 1
                # 传入补充信息
                ch, info = now_choice[labels.index(response[0].lower())]
                history.append(f"User: {ch}: {info}")
                response = self.get_after_api_response("\n".join(history))
                # print(f"Response after: {response}")
                history.append(f"Model: {response}")
                print(history)
            print(f"Response final: {response}")    
            turns.append(cnt)
            complete_history.append('\n'.join(history))
        return predictions, turns, complete_history

    def get_api_response(self, text):
        response = self.client.chat.completions.create(
            model = f"{self.model}",
            messages=[{"role": "system", 
                       "content": '''
                                    Based on your stored knowledge, without performing external searches, answer the following question.
                                    - If you are confident in your answer, provide it directly, and answer with the beginning: "The answer is ..."
                                    - If you have no idea, please answer: "I don't know."
                                    - If you're unsure about which entity I refer to, select the most helpful option from the options I provide. I will feedback corresponding details to assist you clarify the entity. 
                                    Notice: 1. The options are only meant to help you recall the relevant entity; the final answer will not be in one of the options. 
                                            2. The additional information will be in the form of A B C D, and you can only choose the corresponding letter. 
                                            3. If you choose an option, please place it at the **beginning** of your response to make it easily identifiable.
                                    
                                    Example:
                                    [User] Question: The birthDate of Ueland is __ . Additional information:\tA. deathDate\tB. birthPlace\tC. deathYear\tD. deathPlace
                                    [Model] A. I am not sure which Ueland you refer to, I need some key information, such as deathDate.
                                    [User] The deathDate is 1927-03-01.
                                    [Model] B. I need more information such as birthPlace.
                                    [User] Akron, Ohio.
                                    [Model] The answer is 1860-10-10.
                                    '''},
                      {"role": "user", 
                       "content": text}],
        )
        return response.choices[0].message.content

    def get_after_api_response(self, question):
        response = self.client.chat.completions.create(
            model = f"{self.model}",
            messages=[ {"role": "system", 
                        "content": "Based on the given context and your stored knowledge, \
                                    without performing external searches, answer the following question."},
                        {"role": "user", "content": question},],
        )
        return response.choices[0].message.content

class BaseEvaluator(object):
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def write_to_file(self, outputList:list, filename:str, mycls="Person"):
        os.makedirs(f"output/evaluation/{mycls}", exist_ok=True)
        with open(f"output/evaluation/{mycls}/{filename}", "w", encoding="utf-8") as f:
            for idx in range(len(outputList)):
                for item in outputList[idx]:
                    f.write(f"{item}\t")
                f.write("\n")

    def eval(self, nameList:list, ansList:list):
        outputList = []
        precision = 0
        for idx in tqdm(range(len(ansList)), desc="Evaluating answers..."):
            name = nameList[idx]; ans = ansList[idx]
            res = self.get_api_response(name, ans)
            precision += 1 if "yes" in res.lower() else 0
            outputList.append(res)
        precision /= len(ansList)

        return precision, outputList


class MultiAnswerEvaluator(BaseEvaluator):
    def __init__(self, client, model):
        super().__init__(client, model)

    def get_api_response(self, name, ans):
        response = self.client.chat.completions.create(
            temperature = 0, 
            model = f"{self.model}",  # 填写需要调用的模型名称
            messages = [{"role": "system",
                         "content": ''' 
                                    You are an evaluator tasked with assessing whether a response demonstrates the recognition of an ambiguous name and provides multiple **distinct** entities or references related to it.

                                    ### Requirements:
                                    1. A correct response must include:
                                    - At least two **potentially distinct** interpretations, references, or areas related to the ambiguous name.
                                    - These interpretations or references do not need to be fully detailed or specific, as long as they represent plausible possibilities or directions.
                                    - If the explanations are variations or cultural extensions of the same concept, they are **not** considered distinct entities.
                                    
                                    2. A response will be evaluated as "No" if:
                                    - It mentions only one entity or reference.
                                    - The response merely describes different aspects of the same entity.

                                    3. Edge Cases:
                                    - If all interpretations are extensions or variations of the same entity (e.g., different cultural depictions of a single goddess), the answer is "No."
                                    - If the response lacks specificity but mentions multiple plausible areas of application, the evaluation should still return "Yes."

                                    ### Output:
                                    If the response successfully identifies multiple distinct entities or references, return "Yes."  
                                    If it does not meet these requirements, return "No."  

                                    **Notice:** Output only "Yes" or "No" based on this evaluation, without any additional explanations.
                                    '''},
                        {"role": "user",
                         "content": f'''
                                    Ambiguous name: {name}
                                    Response: {ans}
                                     '''}]
        )
        return response.choices[0].message.content


class UniqueAnswerEvaluator(BaseEvaluator):
    def __init__(self, client, model):
        super().__init__(client, model)
    
    def get_api_response(self, name, ans):
        response = self.client.chat.completions.create(
            temperature=0, 
            model=f"{self.model}",  # 填写需要调用的模型名称
            messages=[{"role": "system",
                    "content": '''
                                You are an evaluator tasked with assessing whether a response provides a unique and accurate answer related to a specific, clearly identified entity.  

                                Here is the process:  
                                1. I will provide you with a specific entity and a corresponding response.  
                                2. Your task is to evaluate whether the response delivers a unique and accurate answer directly related to the given entity.  
                                3. A correct response must meet the following criteria:  
                                    - The response focuses solely on the given entity without introducing unrelated information.  
                                    - The response provides unique and relevant details about the entity.  
                                4. If the response satisfies these criteria, return "Yes".  
                                5. If the response fails to focus on the given entity, or fails to provide a unique and relevant answer, return "No".  

                                **Notice**: Output only "Yes" or "No" based on this evaluation, without any additional explanations.
                                '''},
                    {"role": "user",
                    "content": f'''
                                Entity: {name}
                                Response: {ans}
                                '''}]
        )
        return response.choices[0].message.content


class EntityExistEvaluator(BaseEvaluator):
    def __init__(self, client, model):
        super().__init__(client, model)

    def get_api_response(self, entity:str, result:str):
        response = self.client.chat.completions.create(
            temperature=0, 
            model=f"{self.model}",  # 填写需要调用的模型名称
            messages=[
                    {"role": "system",
                    "content": '''
                                I will give you:
                                    Entity: A single item that needs to be checked.
                                    Results List: A list of items obtained from a query.
                                Your task is to evaluate whether the <Entity> exists in the <Results List> and return:
                                    * Yes: If the Entity is present in the Results List.
                                    * No: If the Entity is not present in the Results List.
                                **Notice**: Output only "Yes" or "No" based on this evaluation, without any additional explanations.
                                '''},
                    {"role": "user",
                    "content": f'''
                                Entity: {entity}
                                Results List: {result}
                                '''}]
        )
        return response.choices[0].message.content
    
    def eval(self, entities:list, results:list):
        bigRecall = 0
        outputList = []
        for idx in tqdm(range(len(entities)), desc="Evaluating answers..."):
            recall = 0
            oneList = []
            entityList = entities[idx] #entity还是一个list
            result = results[idx]
            for entity in entityList:
                res = self.get_api_response(entity, result)
                recall += 1 if res.lower() == "yes" else 0
                oneList.append(res)
            recall /= len(entityList)
            bigRecall += recall
            outputList.append('\t'.join(oneList))
        bigRecall /= len(entities)
        return bigRecall, outputList


class BinaryJudgeEvaluator(BaseEvaluator):
    def __init__(self, client, model):
        super().__init__(client, model)
    
    def eval(self, answers:list, predictions:list):
        predictions = list(np.array(predictions).reshape(-1, 2))
        answers = list(np.array(answers).reshape(-1, 2))
        pos_precision, neg_precision, cross_precision, all_precision = 0, 0, 0, 0
        for idx in tqdm(range(len(answers)), desc="Evaluating answers..."):
            ans1, ans2 = answers[idx][0], answers[idx][1]
            pred1, pred2 = predictions[idx][0], predictions[idx][1]
            pred1 = "yes" if "yes" in pred1.lower() else "no"
            pred2 = "yes" if "yes" in pred2.lower() else "no"
            pos_precision += 1 if ans1.lower() == pred1.lower() else 0
            neg_precision += 1 if ans2.lower() == pred2.lower() else 0
            cross_precision += 1 if ans1.lower() == pred1.lower() and ans2.lower() == pred2.lower() else 0
            all_precision += 1 if ans1.lower() == pred1.lower() else 0
            all_precision += 1 if ans2.lower() == pred2.lower() else 0
        pos_precision /= len(answers)
        neg_precision /= len(answers)
        cross_precision /= len(answers)
        all_precision /= 2 * len(answers)
        return pos_precision, neg_precision, cross_precision, all_precision


class MatchRateEvaluator(BaseEvaluator):
    def __init__(self, client, model):
        super().__init__(client, model)
    
    def eval(self, answers:list, predictions:list):
        precision = 0
        for idx in tqdm(range(len(answers)), desc="Evaluating answers..."):
            # ans = answers[idx].strip().strip(',.').lower() ## add
            # pred = predictions[idx].strip().strip(',.').lower()[0] ## add
            # precision += 1 if ans.lower() == pred.lower() else 0
            ans = answers[idx].strip().strip(',.')
            match = re.search(r"\b([A-Z])(?:\.\s?.*)?$",predictions[idx].strip())
            precision += 1 if match and match.group(1) == ans else 0
        precision /= len(answers)
        return precision


class InfoCompleteEvaluator(object):
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def get_api_response(self, text):
        q, llm, ans = text["q"], text["llm"], text["ans"]
        response = self.client.chat.completions.create(
            temperature = 0, 
            model = f"{self.model}",  # 填写需要调用的模型名称
            messages=[{"role": "system", 
                       "content": "Please act as a professional answer evaluation expert and compare the following two answers."},
                       {"role": "user", 
                       "content": f'''
                                    Question: {q}
                                    Model's answer: {llm}
                                    Standard answer: {ans}.

                                    Evaluation Criteria:
                                        - If the model's answer is exactly the same as the standard answer, respond with "YES".
                                        - If the model's answer is phrased differently but overall conveys the same meaning as the standard answer, respond with "YES".
                                        - If the model's answer significantly differs from the standard answer, respond with "NO".

                                    Your Task:
                                        - Compare the model's answer and the standard answer based on the above evaluation criteria
                                        - Provide a clear and concise evaluation: respond with "YES" or "NO\" based on the criteria.
                                        
                                    NOTICE, Only respond with either "YES" or "NO", without any additional explanation.
                                    '''
                        },
                    ],
        )
        return response.choices[0].message.content

    def eval(self, answers:list, predictions:list, questions:list):
        precision = 0
        results = []
        for idx in tqdm(range(len(answers)), desc="Evaluating answers..."):
            q, llm, ans = questions[idx], predictions[idx], answers[idx]
            res = self.get_api_response({"q": q, "llm": llm, "ans": ans})
            results.append(res)
            precision += 1 if res.lower() == "yes" else 0
        precision /= len(answers)
        return precision, results
    
    def eval_item(self, answer:str, prediction:str, question:str):
        res = self.get_api_response({"q": question, "llm": prediction, "ans": answer})
        precision = 1 if res.lower() == "yes" else 0
        return precision, res

    def overall_eval(self, baseResults:list, upperResults:list):
        ac, base = 0, 0
        for idx in range(len(baseResults)):
            if baseResults[idx].lower() == "yes" and upperResults[idx].lower() == "yes":
                ac += 1
            if baseResults[idx].lower() == "yes": base += 1
        if base == 0: return 0
        return ac / base
     

class ContextualBot(object):
    def __init__(self, genClient, genName, evalClient, evalName):
        self.generator = InfoCompleteGenerator(genClient, genName)
        self.evaluator = InfoCompleteEvaluator(evalClient, evalName)
        self.before_precision = 0
        self.after_precision = 0
        self.avg_turn = 0
        self.cnt_turn = 0
        self.predictions = []
    
    def chat(self, questions, addtionals, answers):
        before_precision, after_precision, avg_turn, cnt_turn = 0, 0, 0, 0
        predictions, precisions = [], []
        for i in tqdm(range(len(questions)), desc="Chatting..."):
            question, answer, addition = questions[i], answers[i], addtionals[i]
            response = self.generator.get_api_response(question)
            ####### 直接加入
            history = [f"User: {question}", f"Model: {response}"]
            precision, _ = self.evaluator.eval_item(answer, response, question)
            # 如果回答正确，就不再继续问
            if precision == 1:
                predictions.append(response)
                before_precision += 1
                precisions.append("YES")
            # 如果回答错误，就补充条件
            else:
                precision = 0 #默认回答错误
                for idx in range(len(addition)):
                    history.append(f"User: The answer seems to be incorrect, I have some additional information: {addition[idx]}")
                    response = self.generator.get_api_response("\n".join(history))
                    history.append(f"Model: {response}")
                    precision, _ = self.evaluator.eval_item(answer, response, question)
                    # 如果获得补充信息后，回答正确，则不用再继续加了
                    if precision == 1: 
                        after_precision += 1
                        cnt_turn += 1
                        avg_turn += idx+1
                        break
                predictions.append("\n".join(history))
                precisions.append("YES" if precision == 1 else "NO")
        
        after_precision = (before_precision + after_precision) / len(questions)
        before_precision /= len(questions)
        avg_turn /= cnt_turn

        return before_precision, after_precision, avg_turn, predictions, precisions