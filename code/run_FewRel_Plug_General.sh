python run.py --config config/FewRel/Plug_General/BERT_5way1shot.config  --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/BERT_5way5shot.config  --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/BERT_10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/BERT_10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin

python run.py --config config/FewRel/Plug_General/LoRA_5way1shot.config  --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/LoRA_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/LoRA_5way5shot.config  --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/LoRA_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/LoRA_10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/LoRA_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/LoRA_10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/LoRA_5way1shot/ckpt.bin

python run.py --config config/FewRel/Plug_General/Adapter_5way1shot.config  --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/Adapter_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/Adapter_5way5shot.config  --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/Adapter_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/Adapter_10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/Adapter_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/Adapter_10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/Adapter_5way1shot/ckpt.bin

python run.py --config config/FewRel/Plug_General/BitFit_5way1shot.config  --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BitFit_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/BitFit_5way5shot.config  --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BitFit_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/BitFit_10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BitFit_5way1shot/ckpt.bin
python run.py --config config/FewRel/Plug_General/BitFit_10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/BaseModel/BitFit_5way1shot/ckpt.bin
