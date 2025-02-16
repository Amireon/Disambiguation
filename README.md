from run_ambiguity import data_scalefrom run_ambiguity import data_scalefrom run_ambiguity import data_scale

# Rethinking the Role of LLMs as Knowledge Bases: Insights from Entity Ambiguity Resolution

## Data
### Extract disambiguation links from DBPedia
#### disambiguations_lang=en.ttl
*  extracted from https://databus.dbpedia.org/dbpedia/generic/disambiguations/2020.11.01
* organized as (disambiguation_head, http://dbpedia.org/ontology/wikiPageDisambiguates, specific_entity). 1 vs N format

    <http://dbpedia.org/resource/100_greatest> ｜ <http://dbpedia.org/ontology/wikiPageDisambiguates> ｜ <http://dbpedia.org/resource/100_Greatest_(TV_series)> .

    <http://dbpedia.org/resource/100_greatest> ｜ <http://dbpedia.org/ontology/wikiPageDisambiguates> ｜ <http://dbpedia.org/resource/100_Greatest_African_Americans> .

#### shuffled_disambiguations_lang=en.ttl
* transformed from <disambiguations_lang=en.ttl>
* random shuffled, not assigned by alphabetic order

#### cleaned_shuffled_disambiguations_lang=en.ttl
* clean_links.py
* delete totally same links
* 1 disambiguation v.s N entities

### Sort links by class
#### {Class}_disambiguation.ttl
* construct_dataset.py
* extracted from cleaned_diffname_shuffled_disambiguations_lang=en.ttl
* the entities in links are assigned to same {Class}
* 2k disambiguation name in total (for further choice)

#### {Class}_triples
* clean_triples.py
* triples of entity in cleaned_diffname_shuffled_disambiguations_lang=en.ttl
* filter some duplicated property (1 property v.s N values)
* filter some useless property, save meaningful ones
* shuffle property
* Note: we manually select meaning propertys and form a dict


## Question Generation

According to the defined question categories, combined with their respective evaluation indicators, we can generate corresponding questions.

### Ambiguity

In the `Ambiguity` question category, we define two evaluation metrics and two corresponding question generation templates.

- `Ambiguity Discover Rate(ADR)`: Please give me some information about [ambiguous name].
- `Ambiguity Recall Rate(ARR)`: List all the specific entities associated with [ambiguous name].

```python
data_scale = 100
cls_list = ["Person", "Place", "Organisation", "Building", "Work", "MultiClass"]

for cls in cls_list:
    question_generator = QuestionGenerator(cls, data_scale)
    
    # multiQA for ADR, and links for ARR
    multiQA, links = question_generator.genq_multi_turn()
```

### Disambiguation

Two evaluation metrics, and two question generation templates:
- `Binary Ambiguation(BA)`: The [predicate] of [ambiguous name] is [object]. Is the [ambiguous name] referring to [entity name]?
- `Ambiguity Match Rate(AMR)`: The [predicate] of [ambiguous name] is [object]. Which one is [ambiguous name] referring to? a.[entity name1] b. c. ...

```python
data_scale = 100
attr_scale = 5
cls_list = ["Person", "Place", "Organisation", "Building", "Work", "MultiClass"]

for one_cls in cls_list:
    question_generator = QuestionGenerator(one_cls, data_scale, attr_scale)
    disambqa = question_generator.genq_binary_judge(n=attr_scale) # for BA
    question_generator.genq_match_rate(disambqa) # for AMR
```


### Contextual Resolution

Two evaluation metrics, but just one question generation template:
- `Ambiguity Completion Rate (ACR)`: The [predicate1] of [ambiguous name] is [object1]. The [predicate2]of [name] is __.
- `Multi-turn Adjustment (MA)`: The [predicate1] of [ambiguous name] is [object1]. The [predicate2] of [name] is __. (Feedback information: The [predicate3] of [name]is [object3], ...)

```python
data_scale = 100
attr_scale = 5
cls_list = ["Person", "Place", "Organisation", "Building", "Work", "MultiClass"]

# for each class, generate 5 questions with 2 choices
for one_cls in cls_list:
    question_generator = QuestionGenerator(one_cls, data_scale, attr_scale)
    question_generator.genq_contextual(5, 2)
```

You can generate all questions at once, so that when evaluating the model, you can directly call the corresponding question set. You can also choose to generate questions as needed when evaluating the model.

## Model Evaluation

### Ambiguity

- `Binary Ambiguation(BA)`: The [predicate] of [ambiguous name] is [object]. Is the [ambiguous name] referring to [entity name]?
- `Ambiguity Match Rate(AMR)`: The [predicate] of [ambiguous name] is [object]. Which one is [ambiguous name] referring to? a.[entity name1] b. c. ...


```python
question_generator = QuestionGenerator(mycls, data_scale)
multiQA, links = question_generator.genq_multi_answer()

## Step 1: ADR
multiGenerator = BaseGenerator(genClient, genName)
multiPredictions = multiGenerator.generate(list(multiQA.values()))
multi_evaluator = MultiAnswerEvaluator(evalClient, evalName)
multiPrecision, multiOutputList = multi_evaluator.eval(list(multiQA.keys()), multiPredictions)
print(f"ADR: {multiPrecision}")

## Step 2: ARR
recallGenerator = MultiAnswerGenerator(genClient, genName)
recallPredictions = recallGenerator.generate(list(multiQA.keys()))
recallEvaluator = EntityExistEvaluator(evalClient, evalName)
ARR, outputList = recallEvaluator.eval(links, recallPredictions)
print(f"ARR: {ARR}")
```

### Disambiguation

- `Binary Ambiguation(BA)`: The [predicate] of [ambiguous name] is [object]. Is the [ambiguous name] referring to [entity name]?
- `Ambiguity Match Rate(AMR)`: The [predicate] of [ambiguous name] is [object]. Which one is [ambiguous name] referring to? a.[entity name1] b. c. ...

```python
## Step 1: BA
with open(f"{input_dir}binary_judge.json", "r", encoding="utf-8") as f:
    disamb2qa = json.load(f)

biGenerator = BinaryJudgeGenerator(genClient, genName) # model generating predictions
biEvaluator = BinaryJudgeEvaluator(evalClient, evalName) # model judging predictions

questions, answers = [], []
for disambEntityUrl, info in disamb2qa.items():
    questions.extend([info["questions"][0][0], info["questions"][1][0]])
    answers.extend([info["questions"][0][1], info["questions"][1][1]])
predictions = biGenerator.generate(questions)
pos_precision, neg_precision, cross_precision, all_precision = biEvaluator.eval(answers, predictions)

print(f"True Postive: {pos_precision}")
print(f"True Negative: {neg_precision}")
print(f"Pair Accuracy: {cross_precision}")
print(f"Accuracy: {all_precision}")

matchGenerator = MatchRateGenerator(genClient, genName) 
matchEvaluator = MatchRateEvaluator(evalClient, evalName)

## Step 2: AMR
with open(f"{input_dir}match_rate.json", "r", encoding="utf-8") as f:
    match_disamb2qa = json.load(f)

questions, answers = [], []
for disambEntityUrl, qa in match_disamb2qa.items():
    questions.append(qa[0]); answers.append(qa[1])
predictions = matchGenerator.generate(questions)
AMR = matchEvaluator.eval(answers, predictions)

print(f"MR: {AMR}")
```

### Contextual Resolution

- `Binary Ambiguation(BA)`: The [predicate] of [ambiguous name] is [object]. Is the [ambiguous name] referring to [entity name]?
- `Ambiguity Match Rate(AMR)`: The [predicate] of [ambiguous name] is [object]. Which one is [ambiguous name] referring to? a.[entity name1] b. c. ...

```python
with open(f"{input_dir}contextual_qa.json", "r", encoding='utf-8') as f:
    disamb2qa = json.load(f)

# Choose a subset of data
disamb2qa = dict(islice(disamb2qa.items(), data_scale))
questions, additionals, answers = [], [], []
for disambEntityUrl in disamb2qa.keys():
    val_dicts = disamb2qa[disambEntityUrl]
    questions.append(val_dicts["qa"][0])
    answers.append(val_dicts["qa"][1])
    additionals.append(val_dicts["additionals"])

contextualBot = ContextualBot(genClient, genName, evalClient, evalName)
acc_before, acc_after, avg_turn, predictions, precisions = contextualBot.chat(questions, additionals, answers)

print(f'''Contextual Resolution,
    "Questions Generated": {len(questions)},
    "Accuracy Before": {acc_before},
    "Accuracy After": {acc_after},
    "Average Turns": {avg_turn}''')
```

