deepspeed --num_gpus=1 BERT_fine_tune.py --model klue/bert-base --batch_size 64 --epoch 10  ## 256 oom 뜸 
# 
deepspeed --num_gpus=1 BERT_fine_tune.py --model skt/kobert-base-v1 --batch_size 64 --epoch 10   ## 128 oom 뜸  

deepspeed --num_gpus=1 BERT_fine_tune.py --model monologg/kobert --batch_size 64 --epoch 10