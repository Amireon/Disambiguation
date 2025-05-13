# EARBench: Evaluating the Entity Ambiguity Resolution Ability of LLMs

## Overview
EARBench is a comprehensive benchmark designed to evaluate the entity ambiguity resolution capabilities of Large Language Models (LLMs). This benchmark covers multiple entity types and various evaluation tasks to provide a thorough assessment of models' ability to handle entity disambiguation in different contexts.

## Dataset Structure
The benchmark consists of six entity types:
- Person
- Building
- Place
- Organisation
- Work
- MultiClass

Each entity type contains multiple evaluation tasks:

| Task Type | Description | Average Data Size |
|-----------|-------------|------------------|
| Ambiguity Detection | Detect ambiguous names and recall corresponding entities | ~1,800 |
| Entity Identification | Identify the correct entity based on relational and attribute information | ~1,800 |
| Contextual Resolution | QA & Multi-turn QA for missing attribute and relation | ~1,800 samples |


## Evaluation Metrics

### Ambiguity Detection
- Evaluates the model's performance in detecting entity ambiguity

1. **ADR (Ambiguity Detection Rate)**
   - Task: Please give me some information about [ambiguous name].
   - Evaluates the model's ability to detect ambiguous entities

2. **ARR (Ambiguity Recall Rate)**
   - Task: Recall all possible entities of [ambiguous name].
   - Measures the model's capability to recall entities

### Entity identification
- Evaluates the model's performance in identifying correct entity

1. **BDR (Binary Dismbiguation Rate)**
   - Task: "The [predicate] of [ambiguous name] is [object]. Is the [ambiguous name] referring to [entity name]?"
   - Metrics (report pair accuracy in paper): 
     - True Positive Rate
     - True Negative Rate
     - Pair Accuracy
     - Overall Accuracy

2. **AMR (Ambiguity Match Rate)**
   - Task: "The [predicate] of [ambiguous name] is [object]. Which one is [ambiguous name] referring to? a.[entity name1] b. c. ..."
   - Evaluates the model's ability to match ambiguous entities with their correct references

### Contextual Resolution
- Evaluates the model's performance in resolving entity ambiguity through multi-turn dialogue
1. **ACR(Ambiguity Completion Rate)**
   * Task: The [predicate1] of [ambiguous name] is [object1]. The [predicate2] of [name] is \_\_.
   * Metrics
      - Accuracy Before Multi-Turn Dialogue
2. **Average Turns**
   * Task: Feedback information: The [predicate3] of [ambiguous name] is [object3]. The [predicate2] of [ambiguous name] is \_\_?
   * Metrics:
      - Accuracy After Multi-Turn Dialogue
      - Average Number of Turns
<!-- 
## Dataset Statistics

### Entity-wise Distribution
| Entity Type | Binary Judge | Contextual QA | Info Complete | Match Rate | Multi-turn |
|-------------|--------------|---------------|---------------|------------|------------|
| Person | 1,925 | 1,001 | 401 | 401 | 2,449 |
| Building | 1,867 | 771 | 309 | 401 | - |
| Place | 1,824 | 1,001 | 401 | 401 | - |
| Organisation | 1,814 | 1,001 | 401 | 401 | - |
| Work | 1,800 | 1,001 | 401 | 401 | - |
| MultiClass | 1,722 | 1,001 | 401 | 401 | - | -->

## Usage
You need to include all ```.py``` first. To evaluate a model on EARBench:

1. **Ambiguity Detection Evaluation**:
```python
question_generator = QuestionGenerator(mycls, data_scale)
multiQA, links = question_generator.genq_multi_answer()

# Step 1: ADR
multiGenerator = BaseGenerator(genClient, genName)
multiPredictions = multiGenerator.generate(list(multiQA.values()))
multi_evaluator = MultiAnswerEvaluator(evalClient, evalName)
multiPrecision, multiOutputList = multi_evaluator.eval(list(multiQA.keys()), multiPredictions)

# Step 2: ARR
recallGenerator = MultiAnswerGenerator(genClient, genName)
recallPredictions = recallGenerator.generate(list(multiQA.keys()))
recallEvaluator = EntityExistEvaluator(evalClient, evalName)
ARR, outputList = recallEvaluator.eval(links, recallPredictions)
```

2. **Entity Identification**:
```python
# Step1: BDR
biGenerator = BinaryJudgeGenerator(genClient, genName)
biEvaluator = BinaryJudgeEvaluator(evalClient, evalName)
predictions = biGenerator.generate(questions)
pos_precision, neg_precision, cross_precision, all_precision = biEvaluator.eval(answers, predictions)
# Step2: MR
matchGenerator = MatchRateGenerator(genClient, genName)
matchEvaluator = MatchRateEvaluator(evalClient, evalName)
predictions = matchGenerator.generate(questions)
MR = matchEvaluator.eval(answers, predictions)
```

3. **Contextual Resolution**:
```python
# ACR & MT
contextualBot = ContextualBot(genClient, genName, evalClient, evalName)
acc_before, acc_after, avg_turn, predictions, precisions = contextualBot.chat(questions, additionals, answers)
```

<!-- ## Citation
If you use EARBench in your research, please cite our work:
```
@misc{earbench2024,
    title={EARBench: Evaluating the Entity Ambiguity Resolution Ability of LLMs},
    author={*},
    year={2025},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/yourusername/EARBench}}
}
``` -->
