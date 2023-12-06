import json

#load data
with open('candidates_okvqa.json') as f:
    answer_candidates = json.load(f)
with open('mscoco_val2014_annotations.json') as f:
    val_datasets_annotations = json.load(f)['annotations']

#organize answer list
val_datasets = []
for val_a in val_datasets_annotations:
    multi_answers = []
    for ans in val_a['answers']:
        multi_answers.append(ans['raw_answer'])
    row = {'question_id': val_a['question_id'], 'direct_answers': multi_answers}
    val_datasets.append(row)

#compute score for a predicted answer
def direct_scores(pred_answer, direct_answers):
    acc_num = 0
    cnt = 0
    for _, answer_id in enumerate(direct_answers):
        if pred_answer == answer_id:
            cnt += 1
    if cnt ==1:
        acc_num = 0.3
    elif cnt == 2:
        acc_num = 0.6
    elif cnt > 2:
        acc_num = 1
    return acc_num

#Calculate the accuracy of the first candidate answer for all samples
acc = 0.0
for single_sample in val_datasets:
    single_sample['DA_candidate'] = [each_answer['answer'] for each_answer in answer_candidates[str(single_sample['question_id'])]]
    score = []
    for i in single_sample['DA_candidate']:
        score.append(direct_scores(i, single_sample['direct_answers']))
    acc += score[0]
print(acc/len(val_datasets))