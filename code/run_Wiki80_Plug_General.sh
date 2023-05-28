python run.py --config config/Wiki80/Plug_General/BERT.config    --gpu 0 --run_mode test \
              --checkpoint ../output/Wiki80/BaseModel/BERT/ckpt.bin
python run.py --config config/Wiki80/Plug_General/LoRA.config    --gpu 0 --run_mode test \
              --checkpoint ../output/Wiki80/BaseModel/LoRA/ckpt.bin
python run.py --config config/Wiki80/Plug_General/Adapter.config --gpu 0 --run_mode test \
              --checkpoint ../output/Wiki80/BaseModel/Adapter/ckpt.bin
python run.py --config config/Wiki80/Plug_General/BitFit.config  --gpu 0 --run_mode test \
              --checkpoint ../output/Wiki80/BaseModel/BitFit/ckpt.bin
