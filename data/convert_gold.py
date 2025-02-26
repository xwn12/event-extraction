import json
json_file = "./data/ent_output/predict/ent_pred_test.json"
def convert_gold(json_file):
    gold_docs = [json.loads(line) for line in open(json_file)]
    for gold in gold_docs:
        ner = gold['ner'][0]
        pred = gold['predicted_ner'][0]
        pred_copy = pred.copy()
        y=0
        for i, ent in enumerate(pred_copy):
            if ent[2] == 'Protein':
                pred.pop(i-y)
                y += 1
        for ent in ner:
            if ent[2] == 'Protein':
                pred.append(ent.copy())
    with open('./data/ent_output/predict/gold/ent_pred_test.json', 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in gold_docs))
convert_gold(json_file)


# def convert_no_val(json_file):
#     gold_docs = [json.loads(line) for line in open(json_file)][0]
#     for gold in gold_docs:
#         rela = gold['relations'][0]
#         rela_copy = rela.copy()
#         y=0
#         for i, rel in enumerate(rela_copy):
#             if rel[4] == 'val_connect':
#                 rela.pop(i-y)
#                 y += 1
#     with open('./data/genia/no_val/test.json', 'w') as f:
#         f.write('\n'.join(json.dumps(doc) for doc in gold_docs))
# convert_no_val(json_file)


f1_file = "./data/ent_output/predict/ent_pred_test.json"
def count_f1(f1_file):
    gold_docs = [json.loads(line) for line in open(f1_file)]
    cor = 0
    gold = 0
    pred = 0
    for docs in gold_docs:
        ner = docs['ner'][0]
        pred_ner = docs['predicted_ner'][0]
        if len(docs['sentences'][0])>128:
            print(docs['doc_key'])
        for a in ner:
            if a[-1] == 'Protein':
                gold += 1
            for b in pred_ner:
                if a == b and a[-1] == 'Protein':
                    cor += 1
        for b in pred_ner:
            if b[-1] == 'Protein':
                pred += 1
    p = cor / pred if cor > 0 else 0.0
    r = cor / gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    print(f1,p,r)
    print(cor,gold,pred)




if __name__ == "__main__":
    # 这里是输出路径
    output_file = './data/ent_output/predict/gold/merge/ent_pred_test.json'
    convert_gold(output_file)
    # 合并句子只有当没有给定结果的时候需要
    # pred_docs = [json.loads(line) for line in open("./data/ent_output/predict/gold/ent_pred_test.json")]
    # tmp_docs = {}
    # docs_map = {'doc_key':0,'sentences':[[]],'ner':[[]],'relations':[[]],'predicted_ner':[[]],'predicted_relations':[[]],'list_length':[[]]}
    # merge_docs = []
    # # 将相同的doc_key归到一类
    # for docs in pred_docs:
    #     doc_key = docs['doc_key']
    #     if doc_key//100 not in tmp_docs:
    #         tmp_docs[doc_key//100] = {'sentences':[],'ner':[],'predicted_ner':[],'list_length':[]}
    #     tmp_docs[doc_key//100]['sentences'].append(docs['sentences'][0].copy())
    #     tmp_docs[doc_key//100]['predicted_ner'].append(docs['predicted_ner'][0].copy())
    #     tmp_docs[doc_key//100]['ner'].append(docs['ner'][0].copy())
    #     tmp_docs[doc_key//100]['list_length'].append(docs['list_length'].copy())
    # # 合并句子
    # for key in tmp_docs:
    #     sentences = tmp_docs[key]['sentences']
    #     predicted_ner = tmp_docs[key]['predicted_ner']
    #     ners = tmp_docs[key]['ner']
    #     list_length = tmp_docs[key]['list_length']
    #     count = 1
    #     state = 0
    #     length = 0
    #     merge_state = 0
    #     for i in range(len(sentences)):
    #         for ner in predicted_ner[i]:
    #             if ner[2] != 'Protein':
    #                 state = 1
    #                 break
    #         if i != len(sentences)-1:
    #             for ner in predicted_ner[i]:
    #                 if ner[2] != 'Protein':
    #                     merge_state = 1
    #                     break
    #         # 为ner加长度
    #         for ner in predicted_ner[i]:
    #             ner[0] = ner[0]+length
    #             ner[1] = ner[1]+length
    #         for ner in ners[i]:
    #             ner[0] = ner[0]+length
    #             ner[1] = ner[1]+length
    #         docs_map['sentences'][0] = docs_map['sentences'][0] + sentences[i].copy()
    #         docs_map['ner'][0] = docs_map['ner'][0] + ners[i].copy()
    #         docs_map['predicted_ner'][0] = docs_map['predicted_ner'][0] + predicted_ner[i].copy()
    #         docs_map['list_length'][0] = docs_map['list_length'][0] + list_length[i].copy()
    #         # if len(docs_map['list_length'][0])>1:
    #         #     if docs_map['list_length'][0][-1] - docs_map['list_length'][0][-2] > 1:
    #         #         merge_state = 0
    #         # 合并句子
    #         if state == 1 and len(docs_map['sentences'][0]) < 64 and i != len(sentences)-1 and merge_state == 1:
    #             length = len(docs_map['sentences'][0])
    #         else:
    #             docs_map['doc_key'] = key*100 + count
    #             count = count+1
    #             merge_docs.append(docs_map.copy())
    #             state = 0
    #             merge_state = 0
    #             docs_map = {'doc_key':0,'sentences':[[]],'ner':[[]],'relations':[[]],'predicted_ner':[[]],'predicted_relations':[[]],'list_length':[[]]}
    #             length = 0

    # with open(output_file, 'w') as f:
    #     f.write('\n'.join(json.dumps(doc) for doc in merge_docs))





