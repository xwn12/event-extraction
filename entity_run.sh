python run_entity.py \
     --do_eval --eval_test \
    --learning_rate=1e-5 --task_learning_rate=5e-4 \
    --train_batch_size=16 \
    --context_window 300 \
    --task genia \
    --data_dir ./data/genia/predict \
    --model scibert_scivocab_uncased \
    --output_dir data/ent_output