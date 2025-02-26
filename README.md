# 事件提取
本项目尝试基于 pure 模型和一个特别关系的设定提出提取完整事件提出新方案，对三个论元以下的嵌套事件、关系提取中的触发器嵌套提取问题以及事件提取的子事件重叠提取问题提出了解决方案。
<img src="./figs/test2.png">

## 介绍
其中其中红色虚线代表 Cause 关系，绿色虚线代表Theme 关系，蓝色实线代表 val_connect 关系。通过选取其中一个嵌套事件，“p300”以 val_connect 关系类型和“recruitment”连接与“recruitment”以 theme 关系类型与“p300”连接完整的确定了这个嵌套子事件，“p300”通过 val_connect 关系类型与“interfere”连接、“interfere”以 theme 关系类型与“recruitment”连接以及嵌套子事件的确定，确定了这个嵌套子事件属于这个嵌套事件，而 Cause 关系类型对应的 Foxp3 只要同时作为“interfer”的cause关系类型的客体以及“p300”的 val_connect 关系类型的客体。
<img src="./figs/rel.jpg">
本文就可以确定这个“Foxp3”是“interfer”的相应的论元。并且确定了他的唯一性。可以看到通过触发器“interfere”和嵌套子事件的论元”p300”唯一的确定这个嵌套事件，得到一个唯一的结果。但是其中存在着一个问题就是每一次的额外关系构建都构成了一个又一个的环，会影响关系提取的结果，所以这里采用 pure 模型的思路独立的进行每一次关系提取，从而保证每一个关系都能有效的提取。
<img src="./figs/rel_table.png">

## 实现过程

### 安装依赖包
安装依赖包:
```
pip install -r requirements.txt
```

### 下载数据集并预处理
在此处下载genia11，处理方式如OneEE:
* genia11: [genia11](https://bionlp-st.dbcls.jp/GE/2011/downloads/).

## 方法实现
.

```bash

# Run the pre-trained entity model, the result will be stored in ${genia11_ent_model}/ent_pred_test.json
python run_entity.py \
    --do_eval --eval_test \
    --context_window 0 \
    --task scierc \
    --data_dir ${genia11_dataset} \
    --model allenai/scibert_scivocab_uncased \
    --output_dir ${genia11_ent_model}

# Run the pre-trained full relation model
python run_relation.py \
  --task scierc \
  --do_eval --eval_test \
  --model allenai/scibert_scivocab_uncased \
  --do_lower_case \
  --context_window 0\
  --max_seq_length 128 \
  --entity_output_dir ${genia11_ent_model} \
  --output_dir ${genia11 _rel_model}
  
# Output end-to-end evaluation results
python run_eval.py --prediction_file ${genia11_rel_model}/predictions.json
```

## 实体提取模型

### 实体提取模型的输入数据的格式

实体模型的输入数据格式为JSONL。输入文件的每一行都包含一个以下格式的文档。
```
{
    "doc_key": 1032887401,
    "sentences": [
        [
            "\n",
            "Recent",
            "studies",
            "have",
            "shown",
            "that",
            "the",
            "non",
            "-",
            "steroidal",
            "anti",
            "-",
            "inflammatory",
            "drugs",
            "(",
            "NSAIDs",
            ")",
            "activate",
            "heat",
            "shock",
            "transcription",
            "factor",
            "(",
            "HSF1",
            ")",
            "from",
            "a",
            "latent",
            "cytoplasmic",
            "form",
            "to",
            "a",
            "nuclear",
            ",",
            "DNA",
            "binding",
            "state",
            "."
        ]
    ],
    "ner": [
        [
            [
                23,
                23,
                "Protein"
            ],
            [
                17,
                17,
                "Positive_regulation"
            ],
            [
                35,
                35,
                "Binding"
            ]
        ]
    ],
    "relations": [
        [
            [
                17,
                17,
                35,
                35,
                "Theme"
            ],
            [
                23,
                23,
                17,
                17,
                "val_connect"
            ],
            [
                35,
                35,
                23,
                23,
                "Theme"
            ],
            [
                23,
                23,
                35,
                35,
                "val_connect"
            ]
        ]
    ]
}
```

### 训练/评估实体提取模型

训练:
```bash
python run_entity.py \
    --do_train --do_eval [--eval_test] \
    --learning_rate=1e-5 --task_learning_rate=5e-4 \
    --train_batch_size=16 \
    --context_window {100} \
    --task {genia11} \
    --data_dir {directory of preprocessed dataset} \
    --model {allenai/scibert_scivocab_uncased} \
    --output_dir {directory of output files}
```
Arguments:
* `--learning_rate`: the learning rate for BERT encoder parameters.
* `--task_learning_rate`: the learning rate for task-specific parameters, i.e., the classifier head after the encoder.
* `--context_window`: the context window size used in the model. `0` means using no contexts. In our cross-sentence entity experiments, we use `--context_window 100` for SciBERT models.
* `--model`: the base transformer model. We use `allenai/scibert_scivocab_uncased` for SciERC.
* `--eval_test`: whether evaluate on the test set or not.

实体模型的预测将被保存在"output_dir"目录下的"ent_pred_dev.json"。如果设置'--eval_test '在测试集中预测"ent_pred_test.json",实体模型的预测文件将成为关系模型的输入文件。

## 关系提取模型
### 关系提取模型的数据输入的格式
关系模型的输入数据格式与实体模型的输入数据格式几乎相同，只是多了一个字段。"predicted_ner"来存储实体模型的预测,用于关系预测，并且通过多余的关系val_connect能够直接得到一个事件。
```bash
{
    "doc_key": 889290301,
    "sentences": [
        [
            "Comparative",
            "analysis",
            "identifies",
            "conserved",
            "tumor",
            "necrosis",
            "factor",
            "receptor",
            "-",
            "associated",
            "factor",
            "3",
            "binding",
            "sites",
            "in",
            "the",
            "human",
            "and",
            "simian",
            "Epstein",
            "-",
            "Barr",
            "virus",
            "oncogene",
            "LMP1",
            "."
        ]
    ],
    "ner": [
        [
            [
                4,
                11,
                "Protein"
            ],
            [
                24,
                24,
                "Protein"
            ]
        ]
    ],
    "relations": [
        [

        ]
    ],
    "list_length": [
        0
    ],
    "predicted_ner": [
        [
            [
                4,
                11,
                "Protein"
            ],
            [
                24,
                24,
                "Protein"
            ]
        ]
    ],
    "predicted_relations": [
        [

        ]
    ]
}
```

### 训练/评估关系提取模型
训练：
```bash
python run_relation.py \
  --task {genia11} \
  --do_train --train_file {path to the training json file of the dataset} \
  --do_eval [--eval_test] [--eval_with_gold] \
  --model {allenai/scibert_scivocab_uncased} \
  --do_lower_case \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --context_window {100} \
  --max_seq_length {128} \
  --entity_output_dir {path to output files of the entity model} \
  --output_dir {directory of output files}
```
Aruguments:
* `--eval_with_gold`: whether evaluate the model with the gold entities provided.
* `--entity_output_dir`: the output directory of the entity model. The prediction files (`ent_pred_dev.json` or `ent_pred_test.json`) of the entity model should be in this directory.

预测结果将存储在"output_dir"文件夹中的"predictions.json"中，格式将与实体模型的输出文件每个文档多一个字段"predicted_relations"。

通过额外的关系唯一确定两个实体从而确定唯一的事件
```bash
python run_eval.py --prediction_file {path to output_dir}/predictions.json
```

### 预训练模型
* [SciBERT (cross, W=300)](https://nlp.cs.princeton.edu/projects/pure/scierc_models/ent-scib-ctx300.zip) (391M): Cross-sentence entity model based on `allenai/scibert_scivocab_uncased`

## 总结
  
针对事件提取如何提取嵌套子事件，验证子事件与嵌套子事件之间的关系提供了新的构建方法。将事件提取模块融入了关系提取模块，加强了关系对之间的联系。证明了两论元的完整事件提取范式，针对嵌套事件提取提出了解决方案。但是三论元及以上还不能完整提取，只能按照 GE11数据集的特点进行特殊事件处理（GE11存在唯一的两个实体作为我们定位整个事件的方式。）通过与现有模型对比，证明了该方法在触发器论元识别上实现了竞争性能，在论元角色识别上实现了最好的效果，在嵌套事件的提取上有一定的性能提升。