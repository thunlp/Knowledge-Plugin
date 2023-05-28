python run.py --config config/FewRel/BaseModel/BERT_5way1shot.config    --gpu 0
python run.py --config config/FewRel/BaseModel/LoRA_5way1shot.config    --gpu 0
python run.py --config config/FewRel/BaseModel/Adapter_5way1shot.config --gpu 0
python run.py --config config/FewRel/BaseModel/BitFit_5way1shot.config  --gpu 0
python run.py --config config/FewRel/BaseModel/RoBERTa_5way1shot.config --gpu 0

python run.py --config config/FewRel/BaseModel/BERT_5way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
python run.py --config config/FewRel/BaseModel/BERT_10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
python run.py --config config/FewRel/BaseModel/BERT_10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin

python run.py --config config/FewRel/BaseModel/LoRA_5way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/LoRA_5way1shot/ckpt.bin
python run.py --config config/FewRel/BaseModel/LoRA_10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/LoRA_5way1shot/ckpt.bin
python run.py --config config/FewRel/BaseModel/LoRA_10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/LoRA_5way1shot/ckpt.bin

python run.py --config config/FewRel/BaseModel/Adapter_5way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/Adapter_5way1shot/ckpt.bin
python run.py --config config/FewRel/BaseModel/Adapter_10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/Adapter_5way1shot/ckpt.bin
python run.py --config config/FewRel/BaseModel/Adapter_10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/Adapter_5way1shot/ckpt.bin

python run.py --config config/FewRel/BaseModel/BitFit_5way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BitFit_5way1shot/ckpt.bin
python run.py --config config/FewRel/BaseModel/BitFit_10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BitFit_5way1shot/ckpt.bin
python run.py --config config/FewRel/BaseModel/BitFit_10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BitFit_5way1shot/ckpt.bin
