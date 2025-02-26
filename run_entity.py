import json
import argparse
import os
import sys
import random
import logging
import time
from tqdm import tqdm
import numpy as np

from shared.data_structures import Dataset
from shared.const import task_ner_labels, get_labelmap
from entity.utils import convert_dataset_to_samples, batchify, NpEncoder
from entity.models import EntityModel

from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import prettytable as pt

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')

def save_model(model, args):
    """
    Save the model to the output directory
    """
    logger.info('Saving model to %s...'%(args.output_dir))
    model_to_save = model.bert_model.module if hasattr(model.bert_model, 'module') else model.bert_model
    model_to_save.save_pretrained(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)

def output_ner_predictions(model, batches, dataset, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    span_hidden_table = {}
    tot_pred_ett = 0
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            off = sample['sent_start_in_doc'] - sample['sent_start']
            k = str(sample['doc_key']) + '-' + str(sample['sentence_ix'])
            ner_result[k] = []
            for span, pred in zip(sample['spans'], preds):
                span_id = '%s::%d::(%d,%d)'%(sample['doc_key'], sample['sentence_ix'], span[0]+off, span[1]+off)
                if pred == 0:
                    continue
                ner_result[k].append([span[0]+off, span[1]+off, ner_id2label[pred]])
            tot_pred_ett += len(ner_result[k])

    logger.info('Total pred entities: %d'%tot_pred_ett)

    js = dataset.js
    for i, doc in enumerate(js):
        doc["predicted_ner"] = []
        doc["predicted_relations"] = []
        for j in range(len(doc["sentences"])):
            k = str(doc['doc_key']) + '-' + str(j)
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
            else:
                logger.info('%s not in NER results!'%k)
                doc["predicted_ner"].append([])
            
            doc["predicted_relations"].append([])

        js[i] = doc

    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc, cls=NpEncoder) for doc in js))

def evaluate(model, batches, tot_gold, ner_id2label):
    """
    Evaluate the entity model
    tot_gold: 黄金实体个数数组
    """
    logger.info('Evaluating...')
    c_time = time.time()
    # 计算预测实体个数
    # cor = 0
    # tot_pred = 0
    # 计算各类实体个数
    cor = [0]*(len(ner_id2label)+1)
    tot_pred = [0]*(len(ner_id2label)+1)
    # 计算正确率
    l_cor = 0
    l_tot = 0

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            for gold, pred in zip(sample['spans_label'], preds):
                if pred != 0 and gold != 0:
                    l_tot += 1
                if pred != 0 and gold != 0 and pred == gold:
                    l_cor += 1
                    cor[pred] += 1
                if pred != 0:
                    tot_pred[pred] += 1
                   
    # acc = l_cor / l_tot
    # logger.info('Accuracy: %5f'%acc)
    # logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d'%(sum(cor), sum(tot_pred), sum(tot_gold)))
    # p = cor / tot_pred if cor > 0 else 0.0
    # r = cor / tot_gold if cor > 0 else 0.0
    # f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    # 每个类别的计数
    p=[0]*(len(ner_id2label)+1)
    r=[0]*(len(ner_id2label)+1)
    f1=[0]*(len(ner_id2label)+1)
    for i in range(1,len(cor)):
        p[i] = cor[i] / tot_pred[i] if cor[i] > 0 else 0.0
        r[i] = cor[i] / tot_gold[i] if cor[i] > 0 else 0.0
        f1[i] = 2 * (p[i] * r[i]) / (p[i] + r[i]) if cor[i] > 0 else 0.0
    pro_p = (cor[1]+cor[2])/(tot_pred[1]+tot_pred[2]) if cor[1] > 0 else 0.0
    pro_r = (cor[1]+cor[2])/(tot_gold[1]+tot_gold[2]) if cor[1] > 0 else 0.0
    pro_f1 = 2 * (pro_p * pro_r) / (pro_p + pro_r) if cor[1] > 0 else 0.0
    all_p = sum(cor) / sum(tot_pred) if sum(cor) > 0 else 0.0
    all_r = sum(cor) / sum(tot_gold) if sum(cor) > 0 else 0.0
    all_f1 = 2 * (all_p * all_r) / (all_p + all_r) if sum(cor) > 0 else 0.0
    # print(sum(cor),sum(tot_pred),sum(tot_gold))
    for i in range(1,len(cor)):
        print(tot_pred[i] )
    # tri_p = (sum(cor)-cor[1]+cor[2]) / (sum(tot_pred)-tot_pred[1]+tot_pred[2]) if sum(cor) > 0 else 0.0
    # tri_r = (sum(cor)-cor[1]+cor[2]) / (sum(tot_gold)-tot_gold[1]+tot_gold[2]) if sum(cor) > 0 else 0.0
    # tri_f1 = 2 * (tri_p * tri_r) / (tri_p + tri_r) if sum(cor) > 0 else 0.0
    table = pt.PrettyTable(['Class','F1', "Precision", "Recall"])
    for i in range(1, len(f1)):
        table.add_row(["{}".format(ner_id2label[i])] + ["{:3.4f}".format(x) for x in [f1[i], p[i], r[i]]])
    table.add_row(["Protein"] + ["{:3.4f}".format(x) for x in [pro_f1, pro_p, pro_r]])
    # table.add_row(["Trigger"] + ["{:3.4f}".format(x) for x in [tri_f1, tri_p, tri_r]])
    table.add_row(["all"] + ["{:3.4f}".format(x) for x in [all_f1, all_p, all_r]])

    logger.info("\n{}".format(table))
    # logger.info('P: %.5f, R: %.5f, F1: %.5f'%(p, r, f1))
    logger.info('Used time: %f'%(time.time()-c_time))
    return f1


def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default=None, required=True, choices=['ace04', 'ace05', 'scierc','genia'])

    parser.add_argument('--data_dir', type=str, default=None, required=True, 
                        help="path to the preprocessed dataset")
    parser.add_argument('--output_dir', type=str, default='entity_output', 
                        help="output directory of the entity model")

    parser.add_argument('--max_span_length', type=int, default=8, 
                        help="spans w/ length up to max_span_length are considered as candidates")
    parser.add_argument('--train_batch_size', type=int, default=32, 
                        help="batch size during training")
    parser.add_argument('--eval_batch_size', type=int, default=32, 
                        help="batch size during inference")
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help="learning rate for the BERT encoder")
    parser.add_argument('--task_learning_rate', type=float, default=1e-4, 
                        help="learning rate for task-specific parameters, i.e., classification head")
    parser.add_argument('--warmup_proportion', type=float, default=0.1, 
                        help="the ratio of the warmup steps to the total steps")
    parser.add_argument('--num_epoch', type=int, default=100, 
                        help="number of the training epochs")
    parser.add_argument('--print_loss_step', type=int, default=100, 
                        help="how often logging the loss value during training")
    parser.add_argument('--eval_per_epoch', type=int, default=1, 
                        help="how often evaluating the trained model on dev set during training")
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")

    parser.add_argument('--do_train', action='store_true', 
                        help="whether to run training")
    parser.add_argument('--train_shuffle', action='store_true',
                        help="whether to train with randomly shuffled data")
    parser.add_argument('--do_eval', action='store_true', 
                        help="whether to run evaluation")
    parser.add_argument('--eval_test', action='store_true', 
                        help="whether to evaluate on test set")
    parser.add_argument('--dev_pred_filename', type=str, default="ent_pred_dev.json", help="the prediction filename for the dev set")
    parser.add_argument('--test_pred_filename', type=str, default="ent_pred_test.json", help="the prediction filename for the test set")

    parser.add_argument('--use_albert', action='store_true', 
                        help="whether to use ALBERT model")
    parser.add_argument('--model', type=str, default='bert-base-uncased', 
                        help="the base model name (a huggingface model)")
    parser.add_argument('--bert_model_dir', type=str, default=None, 
                        help="the base model directory")

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--context_window', type=int, required=True, default=None, 
                        help="the context window size W for the entity model")

    args = parser.parse_args()
    args.train_data = os.path.join(args.data_dir, 'train.json')
    args.dev_data = os.path.join(args.data_dir, 'dev.json')
    args.test_data = os.path.join(args.data_dir, 'test.json')


    setseed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))

    logger.info(sys.argv)
    logger.info(args)
    
    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])
    # ner实体类型数量
    num_ner_labels = len(task_ner_labels[args.task]) + 1
    model = EntityModel(args, num_ner_labels=num_ner_labels)
    # 训练集取样
    dev_data = Dataset(args.dev_data)
    dev_samples, dev_ner = convert_dataset_to_samples(dev_data, args.max_span_length, ner_label2id=ner_label2id, context_window=args.context_window,num_ner_labels=num_ner_labels)
    dev_batches = batchify(dev_samples, args.eval_batch_size)

    if args.do_train:
        train_data = Dataset(args.train_data)
        train_samples, train_ner = convert_dataset_to_samples(train_data, args.max_span_length, ner_label2id=ner_label2id, context_window=args.context_window,num_ner_labels=num_ner_labels)
        train_batches = batchify(train_samples, args.train_batch_size)
        best_result = 0.0

        param_optimizer = list(model.bert_model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                if 'bert' in n]},
            {'params': [p for n, p in param_optimizer
                if 'bert' not in n], 'lr': args.task_learning_rate}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=not(args.bertadam))
        t_total = len(train_batches) * args.num_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup_proportion), t_total)
        
        tr_loss = 0
        tr_examples = 0
        global_step = 0
        eval_step = len(train_batches) // args.eval_per_epoch
        for _ in tqdm(range(args.num_epoch)):
            if args.train_shuffle:
                random.shuffle(train_batches)
            for i in tqdm(range(len(train_batches))):
                output_dict = model.run_batch(train_batches[i], training=True)
                loss = output_dict['ner_loss']
                loss.backward()

                tr_loss += loss.item()
                tr_examples += len(train_batches[i])
                global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % args.print_loss_step == 0:
                    logger.info('Epoch=%d, iter=%d, loss=%.5f'%(_, i, tr_loss / tr_examples))
                    tr_loss = 0
                    tr_examples = 0

                if global_step % eval_step == 0:
                    f1 = evaluate(model, dev_batches, dev_ner, ner_id2label)
                    if f1 > best_result:
                        best_result = f1
                        logger.info('!!! Best valid (epoch=%d): %.2f' % (_, f1*100))
                        save_model(model, args)

    if args.do_eval:
        args.bert_model_dir = args.output_dir
        model = EntityModel(args, num_ner_labels=num_ner_labels)
        if args.eval_test:
            test_data = Dataset(args.test_data)
            prediction_file = os.path.join(args.output_dir, args.test_pred_filename)
        else:
            test_data = Dataset(args.dev_data)
            prediction_file = os.path.join(args.output_dir, args.dev_pred_filename)
        test_samples, test_ner = convert_dataset_to_samples(test_data, args.max_span_length, ner_label2id=ner_label2id, context_window=args.context_window,num_ner_labels=num_ner_labels)
        test_batches = batchify(test_samples, args.eval_batch_size)
        evaluate(model, test_batches, test_ner, ner_id2label)
        output_ner_predictions(model, test_batches, test_data, output_file=prediction_file)
