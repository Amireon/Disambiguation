import os
import re
import json
import random
import collections

########## Params ##########

### Random Seed
SEED = 42
random.seed(SEED)

### Data
folder = "data"
output = "output"

class2property = {"album":["artist", "genre", "length", "prevTitle", "producer", "recorded"]}
property2template = {"recorded": "recorded time",
                     "released": "released time",
                     "firstAired": "first aired time",
                     "lastAired": "last aired time",
                     "numEpisodes": "number of episodes",
                     }

### Valid Property
propertyDict = {"Person":[
                        "birthDate", "birthYear", "deathDate", "deathYear",  
                        "finalyear", "finaldate", "debutyear", "termStart", "termEnd", "activeYearsStartYear", "activeYearsEndYear", "termPeriod",  
                        "birthPlace", "deathPlace", "restingplace",  
                        "country", "nationality", "stateOfOrigin", "province", "religion",  
                        "hometown", "residence",  
                        "position", "occupation", "employer", "currentTeam", "debutfor", "debutteam", "debutleague", "statleague",  
                        "sport", "youthclubs", "club", "collegeteam", "weightLb", "height",  
                        "highschool", "college", "almaMater",  
                        "party", "parliament",  
                        "predecessor", "successor",  
                        "spouse", "parent", "mother", "father", "child",  
                        "genre", "instrument"
                        ]
                }

############################

# starring: is starring
# 处理 val 为空

# load attribute from "property" file
def load_attribute(filename="data/clean_property"):
    predList = []
    with open(filename, "r", encoding="utf-8") as f:
        propertys = f.readlines()
    for i in range(len(propertys)):
        _, pred, _ = propertys[i].strip("\n").split("\t")
        editPred = pred.strip('>').split("/")[-1]
        if editPred not in predList: predList.append(editPred)
    return predList

# 双层字典嵌套
# {disambiguationLink: {entity: [(pred, obj), ...]}}
def load_triple_from_links(tripleFile="data/triples", 
                           linkFile="data/Work_disambiguations.ttl"):
    
    # 循环一次links，读入link中的头和尾实体
    with open(linkFile, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rule = "<(.*?)>.*?<(.*?)>.*?"
    objList = collections.defaultdict(set)
    # 只保留<>内实体的url
    for line in lines:
        m = re.match(rule, line)
        subj, obj = m.group(1), m.group(2)
        objList[obj] = set()
    
    # 循环一次triple文件，读入所有三元组
    rule = "<(.*?)>\t<(.*?)>\t<(.*?)>\n"
    with open(tripleFile, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        m = re.match(rule, line)
        subj, pred, obj = m.group(1), m.group(2), m.group(3)
        pred = transform_uri(pred).strip()
        obj = transform_uri(obj).strip()
        # obj 可能是空值
        if obj == "": continue
        objList[subj].add((pred, obj))
    
    # 再循环一次links，将name和消歧义实体对应起来
    # 双层字典：{disambiguationLink: {entity: [(pred, obj), ...]}}
    with open(linkFile, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rule = "<(.*?)>.*?<(.*?)>.*?"
    name2entityDict = collections.defaultdict(dict)
    for line in lines:
        m = re.match(rule, line)
        subj, obj = m.group(1), m.group(2)
        name2entityDict[subj][obj] = objList[obj]

    return name2entityDict

def transform_uri(uri):
    return uri.strip('<>').split("/")[-1].replace('_', ' ')

def transform_passive(pred):
    # ["recorded", "released", "firstAired", "lastAired", "founded", "published"]
    if pred[-2:].lower() == "ed":
        return " is " + pred + " at "
    # ["written"]
    elif pred[-2:].lower() == "en":
        return " is " + pred + " by "
    elif pred[-2:].lower() == "by":
        return " is " + pred + " "

def write_qa(entity2qa:list, outputFile="data/q1.txt"):
    with open(outputFile, "w", encoding="utf-8") as f:
        for entity, question, answer in entity2qa:
            f.write(f"{entity}\t{question}\t{answer}\n")

def passive_pred(pred):
    end = pred[-2:].lower()
    if end in ["ed", "en", "by"]:
        return True

def active_pred(pred):
    if pred[-3:].lower() == "ing": return True
    return False


class QuestionGenerator(object):
    def __init__(self, mycls, ent_scale, attr_scale=1):
        '''
        params:
            mycls: str, The category.
            ent_scale: int, The scale of entities for the given category (mycls).
            attr_scale: int, The scale of attributes of one entity.
            all: bool, Whether to retain all attributes.
        '''
        self.mycls = mycls
        self.ent_scale = ent_scale
        self.attr_scale = attr_scale

        self.name2objDict = self.start(mycls, ent_scale)

    def random_select(self, objs:list, n=1):
        '''
        func: 
            Randomly select an entity and its triples.
        params:
            objs: [(objEntityUrl, triples), ...]
            n: int, how many attribute to use
        return:
            objEntityUrl: selected entity
            triples: triples of selected entity
        '''
        nflag = False
        for _, triples in objs:
            if len(triples) >= n:
                nflag = True
                break
        if not nflag: return None, None
        while True:
            objEntityUrl, triples = random.choice(objs)
            if len(triples) >= n:
                return objEntityUrl, triples[:n]

    def orderby_select(self, objs:list, n=1):
        '''
        func: 
            Orderly select an entity and its triples.
        '''
        for objEntityUrl, triples in objs:
            if len(triples) >= n:
                return objEntityUrl, triples[:n]
        return None, None

    def genq_info_question(self, disambEntityName, pred):
        def passive_pred(pred):
            end = pred[-2:].lower()
            if end in ["ed", "en", "by"]:
                return True
        def transform_passive(pred):
            if pred[-2:].lower() == "ed":
                return " is " + pred + " at "
            # ["written"]
            elif pred[-2:].lower() == "en":
                return " is " + pred + " by "
            elif pred[-2:].lower() == "by":
                return " is " + pred + " "
        def active_pred(pred):
            if pred[-3:].lower() == "ing": return True
            return False

        if passive_pred(pred):
            question = disambEntityName + transform_passive(pred) + "__ ."
        elif active_pred(pred):
            question = disambEntityName + " is " + pred + " __ ."
        else: # 名词 + Of
            question = f"The {pred} of {disambEntityName} is __ ."   
        
        return question

    def start(self, mycls, ent_scale):
        tripleFile = f"{folder}/triples/{mycls}_triples"
        linkFile = f"{folder}/links/{mycls}_disambiguations.ttl"
        
        # 加载三元组
        name2entityDict = load_triple_from_links(tripleFile,linkFile)

        # 获取前 ent_scale 个实体
        count = 0
        name2objDict = collections.defaultdict(dict)
        for subj, entityList in name2entityDict.items():
            objList = {}
            if count >= ent_scale: break
            for obj, triples in entityList.items():
                # 将 set 转换为 list 以便随机采样
                triples_list = list(set(list(triples)))
                # 如果 triples_list 的长度小于attr_scale，则跳过
                # 可能造成某个实体没有问题。。。
                if len(triples_list) < self.attr_scale: continue
                objList[obj] = triples_list
            # 存在多个歧义实体，但是有的歧义实体没有有效的三元组
            if len(objList) <= 1: continue
            name2objDict[subj] = objList
            count += 1

        return name2objDict

    def genq_uniq_answer(self, file=True):
        '''
        func:
            Q: Please give me some information about The Beast Within(film).
            A: It's ...
        return:
            disamb2qa: {disambEntityUrl: question} 
        '''
        name2objDict = self.name2objDict
        mycls = self.mycls
        disamb2qa = {}
        for disambEntityUrl, _ in name2objDict.items():
            objEntityUrl, _ = self.random_select(list(name2objDict[disambEntityUrl].items()))
            if objEntityUrl == None: continue
            objEntityName = transform_uri(objEntityUrl)
            question = f"Please give me some information about {objEntityName}."
            disamb2qa[objEntityName] = question
        
        if file:
            os.makedirs(f"{output}/question/{mycls}", exist_ok=True)
            with open(f"{output}/question/{mycls}/unique_answer.txt", "w", encoding="utf-8") as f:
                for objEntityName, question in disamb2qa.items():
                    f.write(f"{objEntityName}\t{question}\n")
        
        return disamb2qa


    def genq_multi_answer(self, file=True):
        '''
        func:
            Q: Please give me some information about The Beast Within.
            A: There are more than one ...
        params:
            name2objDict: {disambEntityUrl: {objEntityUrl: [(pred, obj), ...]}}
            file: bool, whether write to file
        return:
            disamb2qa: {disambEntityUrl: question} 
        '''
        name2objDict = self.name2objDict
        mycls = self.mycls
        disamb2qa = {}
        links = []
        for disambEntityUrl, _ in name2objDict.items():
            disambEntityName = transform_uri(disambEntityUrl)
            objEntityUrls = list(name2objDict[disambEntityUrl].keys())
            objEntityNames = [transform_uri(obj) for obj in objEntityUrls]
            links.append(objEntityNames)
            # if "(disambiguation)" in disambEntityName: 
            #     disambEntityName = disambEntityName.split("(")[0].strip()
            question = f"Please give me some information about {disambEntityName}."
            disamb2qa[disambEntityName] = question

        if file:
            os.makedirs(f"{output}/question/{mycls}", exist_ok=True)
            with open(f"{output}/question/{mycls}/multi_answer.txt", "w", encoding="utf-8") as f:
                for disambEntityName, question in disamb2qa.items():
                    f.write(f"{disambEntityName}\t{question}\n")
        
        return disamb2qa, links

    def genq_binary_judge(self, n=2, file=True):
        '''
        func:
            Q: The writer of The Beast Within is Tom Holland. Is "The Beast Within" referring to "The Beast Within (film)"?
        params:
            name2objDict: {disambEntityUrl: {objEntityUrl: [(pred, obj), ...]}}
            n: int, how many attribute to use
            file: bool, whether write to file
        return:
            disamb2qa: {disambEntityUrl: 
                            "chosen_entity": objEntityUrl,
                            "candidate_entity": [entityUrl1, ..., entityUrln]
                            "questions":[(postive_question, "Yes"),(negative_question, "No")],
                        } 
        '''                
        name2objDict = self.name2objDict
        mycls = self.mycls
        disamb2qa = collections.defaultdict(dict)

        for disambEntityUrl, _ in name2objDict.items():
            disambEntityName = transform_uri(disambEntityUrl)
            objEntityUrl, triples = self.random_select(list(name2objDict[disambEntityUrl].items()), n)
            if objEntityUrl == None: continue
            candidate_entity = [entity for entity in name2objDict[disambEntityUrl].keys()]
            other_candidate_entity = [entity for entity in name2objDict[disambEntityUrl].keys() if entity != objEntityUrl]
            negative_entity = random.choice(other_candidate_entity)
            objEntityName, negEntityName = transform_uri(objEntityUrl), transform_uri(negative_entity)
            for idx, (pred, val) in enumerate(triples):
                if idx == 0:
                    question = f"The {pred} of {disambEntityName} is {val}"
                else:
                    question += f", and {pred} is {val}"
            positive_question = question + f". Is the {disambEntityName} referring to {objEntityName}?"
            negative_question = question + f". Is the {disambEntityName} referring to {negEntityName}?"
            disamb2qa[disambEntityUrl]["chosen_entity"] = objEntityUrl
            disamb2qa[disambEntityUrl]["candidate_entity"] = candidate_entity
            disamb2qa[disambEntityUrl]["questions"] = [(positive_question, "Yes"), (negative_question, "No")]
        
        if file:
            os.makedirs(f"{output}/question/{mycls}", exist_ok=True)
            with open(f"{output}/question/{mycls}/binary_judge.json", "w", encoding="utf-8") as f:
                json.dump(disamb2qa, f, ensure_ascii=False, indent=4)
        return disamb2qa

    def genq_match_rate(self, disamb2Entity, n=2, file=True):
        '''
        func:
            Q: The writer of The Beast Within is Tom Holland. Is The Beast Within: a. The Beast Within (novel)  b. The Beast Within (film)
        params:
            disamb2Entity:{  
                disambEntityUrl:{ 
                    "chosen_entity": objEntityUrl,
                    "candidate_entity": [entityUrl1, ..., entityUrln]
                }
            }
            n: int, how many attribute to use
            file: bool, whether write to file
        return:
            disamb2qa: {disambEntityUrl: (question, answer)}
        '''
        name2objDict = self.name2objDict
        mycls = self.mycls
        disamb2qa = {}
        
        for disambEntityUrl, objs in name2objDict.items():
            disambEntityName = transform_uri(disambEntityUrl)
            if disambEntityUrl not in disamb2Entity: continue
            objEntityUrl = disamb2Entity[disambEntityUrl]["chosen_entity"]
            candidateEntity = disamb2Entity[disambEntityUrl]["candidate_entity"]
            candidateEntityName = [transform_uri(entity) for entity in candidateEntity]

            triples = objs[objEntityUrl][:n]
            for idx, (pred, val) in enumerate(triples):
                if idx == 0:
                    question = f"The {pred} of \"{disambEntityName}\" is {val}"
                else:
                    question += f", and {pred} is {val}"

            question += f". Which one is the \"{disambEntityName}\" referring to?"
            question += ''.join([f" {chr(65 + i)}. {name}" for i, name in enumerate(candidateEntityName)])
            answer = chr(65 + candidateEntity.index(objEntityUrl))
            disamb2qa[disambEntityUrl] = (question, answer)

        if file:
            os.makedirs(f"{output}/question/{mycls}", exist_ok=True)
            with open(f"{output}/question/{mycls}/match_rate.json", "w", encoding="utf-8") as f:
                json.dump(disamb2qa, f, ensure_ascii=False, indent=4)        
        return disamb2qa

    def genq_info_complete_identifer(self, n=5, file=True):
        '''
        func:
            Q: The writer of the Beast Within(film) is ___ .
        params:
            name2objDict: {disambEntityUrl: {objEntityUrl: [(pred, obj), ...]}}
            n: int, how many attribute to use
            file: bool, whether write to file
        return:
            disamb2qa: {disambEntityUrl: (question, answer)} 
        '''
        name2objDict = self.name2objDict
        mycls = self.mycls
        disamb2qa = {}
        for disambEntityUrl, objDict in name2objDict.items():
            disambEntityName = transform_uri(disambEntityUrl)
            objEntityUrl, triples = self.random_select(list(objDict.items()), n)
            if objEntityUrl == None: continue
            entityName = transform_uri(objEntityUrl)
            # 取第一个（属性，值）对作为考察
            pred, val = triples[0]
            question = self.genq_info_question(entityName, pred)
            disamb2qa[disambEntityUrl] = {"objEntityUrl": objEntityUrl, 
                                          "triples": triples,
                                          "qa": (question, val)}    
        
        if file:
            os.makedirs(f"{output}/question/{mycls}", exist_ok=True)
            with open(f"{output}/question/{mycls}/info_complete_base.txt", "w", encoding="utf-8") as f:
                for disambEntityUrl in disamb2qa.keys():
                    (question, answer) = disamb2qa[disambEntityUrl]["qa"]
                    f.write(f"{question}\t{answer}\n")
        return disamb2qa    
 
    def genq_info_complete_without_identifer(self, baseDict, n=2, file=True):
        '''
        func:
            Q: The writer of The Beast Within is Tom Holland. What's the release date of The Beast Within?
        params:
            baseDict: {disambEntityUrl:
                        {"objEntityUrl": objEntityUrl,
                        "triples": triples,
                        "qa": (question, answer)}}
            n: number of addtional information
            file: bool, whether write to file
        return:
            disamb2qa: {disambEntityUrl: (question, answer)} 
        '''
        name2objDict = self.name2objDict
        mycls = self.mycls
        disamb2qa = {}
        for disambEntityUrl, _ in name2objDict.items():
            disambEntityName = transform_uri(disambEntityUrl)
            if disambEntityUrl not in baseDict: continue
            objEntityUrl = baseDict[disambEntityUrl]["objEntityUrl"]
            triples = baseDict[disambEntityUrl]["triples"]
            if objEntityUrl == None: continue
            # 取第一个（属性，值）对作为考察
            pred0, val0 = triples[0]
            # 取之后的（属性，值）作为补充信息
            infoTriples = triples[1:1+n]
            for idx, (pred, val) in enumerate(infoTriples):
                if idx == 0:
                    question = f"The {pred} of {disambEntityName} is {val}"
                else:
                    question += f", and {pred} is {val}"
            
            question += '. ' + self.genq_info_question(disambEntityName, pred0)
            disamb2qa[disambEntityUrl] = (question, val0)
        
        if file:
            os.makedirs(f"{output}/question/{mycls}", exist_ok=True)
            with open(f"{output}/question/{mycls}/info_complete.json", "w", encoding="utf-8") as f:
                json.dump(disamb2qa, f, ensure_ascii=False, indent=4)
            # with open(f"{output}/question/{mycls}/info_complete_upper.txt", "w", encoding="utf-8") as f:
            #     for disambEntityUrl, (question, answer) in disamb2qa.items():
            #         f.write(f"{question}\t{answer}\n")
        return disamb2qa    
 
    def genq_multi_turn(self, n=5, file=True):
        '''
        func:
            Q: Question: The writer of The Beast Within is ___. Additional information: <a>type <b>release-date <c>genre <d>length 
        params:
            n: int, number of choices of attributes
            file: bool, whether write to file
        return:
            disamb2qa: {disambEntityUrl: 
                            "qa":(question, answer),
                            "choices"[(choice, answer), ...])} 
        '''
        name2objDict = self.name2objDict
        mycls = self.mycls
        disamb2qa = collections.defaultdict(dict)
        for disambEntityUrl, objDict in name2objDict.items():
            disambEntityName = transform_uri(disambEntityUrl)
            objEntityUrl, triples = self.random_select(list(objDict.items()), n)
            if objEntityUrl == None: continue
            # 取第一个（属性，值）对作为考察
            pred, ans = triples[0]
            # 剩余四个作为选项
            question = "Question: "+ self.genq_info_question(disambEntityName, pred)
            choices = []
            question += " Additional information:"
            for i, (pred, val) in enumerate(triples[1:]):
                question += f"\t{chr(65+i)}. {pred}"
                choices.append((pred,val))
            disamb2qa[disambEntityUrl]["qa"] = (question, ans)
            disamb2qa[disambEntityUrl]["choices"] = choices
        
        if file:
            os.makedirs(f"{output}/question/{mycls}", exist_ok=True)
            with open(f"{output}/question/{mycls}/multi_turn.json", "w", encoding="utf-8") as f:
                json.dump(disamb2qa, f, ensure_ascii=False, indent=4)        

        return disamb2qa    

    def genq_contextual(self, n=5, info=2, file=True):
        '''
        params:
            n: total number of attribute pairs
            info: number of contextual pairs
        return:
            disamb2qa: {disambEntityUrl: 
                            "qa": (question, answer),
                            "additionals": [(str)pair sentence, ...]
                        }
        '''
        name2objDict = self.name2objDict
        mycls = self.mycls
        disamb2qa = collections.defaultdict(dict)

        for disambEntityUrl, _ in name2objDict.items():
            disambEntityName = transform_uri(disambEntityUrl)
            objEntityUrl, triples = self.random_select(list(name2objDict[disambEntityUrl].items()), n)
            entityName = transform_uri(objEntityUrl)
            if objEntityUrl == None: continue
            # 取第一个（属性，值）对作为考察
            pred0, val0 = triples[0]
            # 取之后info个（属性，值）作为上下文信息
            infoTriples = triples[1:1+info]
            for idx, (pred, val) in enumerate(infoTriples):
                if idx == 0:
                    question = f"The {pred} of {disambEntityName} is {val}"
                else:
                    question += f", and {pred} is {val}"
            # 取剩下所有（属性，值）作为属性信息
            addList = []
            addTriples = triples[1+info:]
            for idx, (pred, val) in enumerate(addTriples):
                addList.append(f"The {pred} of {disambEntityName} is {val}.")
            question += '. ' + self.genq_info_question(disambEntityName, pred0)
            disamb2qa[disambEntityUrl]["qa"] = (question, val0)
            disamb2qa[disambEntityUrl]["additionals"] = addList
        
        if file:
            os.makedirs(f"{output}/question/{mycls}", exist_ok=True)
            with open(f"{output}/question/{mycls}/contextual_qa.json", "w", encoding="utf-8") as f:
                json.dump(disamb2qa, f, ensure_ascii=False, indent=4)        

        return disamb2qa


if __name__ == "__main__":
    data_scale = 100
    attr_scale = 5
    cls_list = ["Person", "Place", "Organisation", "Building", "Work", "MultiClass"]

    ## generate questions for disambiguation
    for one_cls in cls_list:
        question_generator = QuestionGenerator(one_cls, data_scale, attr_scale)
        disambqa = question_generator.genq_binary_judge(n=attr_scale)
        question_generator.genq_match_rate(disambqa)
    
    ## for contextual
    for one_cls in cls_list:
        question_generator = QuestionGenerator(one_cls, data_scale, attr_scale)
        question_generator.genq_contextual(5,2)
    