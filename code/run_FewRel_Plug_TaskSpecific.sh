python run.py --config config/FewRel/Plug_TaskSpecific/5way1shot.config  --gpu 0 --run_mode map-tuning \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
python convert.py --path ../output/FewRel/Plug_TaskSpecific/5way1shot
python run.py --config config/FewRel/Plug_TaskSpecific/5way5shot.config  --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_TaskSpecific/10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_TaskSpecific/10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
