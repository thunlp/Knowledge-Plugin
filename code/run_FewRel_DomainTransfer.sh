python run.py --config config/FewRel_DomainTransfer/BERT/5way1shot.config  --gpu 0 --run_mode map-tuning \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
python convert.py --path ../output/FewRel_DomainTransfer/BERT/5way1shot
python run.py --config config/FewRel_DomainTransfer/BERT/5way5shot.config  --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
python run.py --config config/FewRel_DomainTransfer/BERT/10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
python run.py --config config/FewRel_DomainTransfer/BERT/10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin

python run.py --config config/FewRel_DomainTransfer/RoBERTa/5way1shot.config  --gpu 0 --run_mode map-tuning \
              --checkpoint ../output/FewRel/BaseModel/RoBERTa_5way1shot/ckpt.bin
python convert.py --path ../output/FewRel_DomainTransfer/RoBERTa/5way1shot
python run.py --config config/FewRel_DomainTransfer/RoBERTa/5way5shot.config  --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/RoBERTa_5way1shot/ckpt.bin
python run.py --config config/FewRel_DomainTransfer/RoBERTa/10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/RoBERTa_5way1shot/ckpt.bin
python run.py --config config/FewRel_DomainTransfer/RoBERTa/10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/RoBERTa_5way1shot/ckpt.bin