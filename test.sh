deepspeed --num_gpus=1 BERT_fine_tune.py --task klue/bert-base --batch_size 128 --epoch 10  ## 256 oom 뜸 
# 
deepspeed --num_gpus=1 BERT_fine_tune.py --task skt/kobert-base-v1 --batch_size 128 --epoch 10   ## 128 oom 뜸  

deepspeed --num_gpus=1 BERT_fine_tune.py --task monologg/kobert --batch_size 128 --epoch 10