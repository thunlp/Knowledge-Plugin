python run.py --config config/WikiET/Plug_General/BERT.config    --gpu 0 --run_mode test \
              --checkpoint ../output/WikiET/BaseModel/BERT/ckpt.bin
python run.py --config config/WikiET/Plug_General/LoRA.config    --gpu 0 --run_mode test \
              --checkpoint ../output/WikiET/BaseModel/LoRA/ckpt.bin
python run.py --config config/WikiET/Plug_General/Adapter.config --gpu 0 --run_mode test \
              --checkpoint ../output/WikiET/BaseModel/Adapter/ckpt.bin
python run.py --config config/WikiET/Plug_General/BitFit.config  --gpu 0 --run_mode test \
              --checkpoint ../output/WikiET/BaseModel/BitFit/ckpt.bin
