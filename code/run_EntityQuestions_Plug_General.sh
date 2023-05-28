python run.py --config config/EntityQuestions/Plug_General/BERT.config    --gpu 0 --run_mode test \
              --checkpoint ../output/EntityQuestions/BaseModel/BERT/ckpt.bin
python run.py --config config/EntityQuestions/Plug_General/LoRA.config    --gpu 0 --run_mode test \
              --checkpoint ../output/EntityQuestions/BaseModel/LoRA/ckpt.bin
python run.py --config config/EntityQuestions/Plug_General/Adapter.config --gpu 0 --run_mode test \
              --checkpoint ../output/EntityQuestions/BaseModel/Adapter/ckpt.bin
python run.py --config config/EntityQuestions/Plug_General/BitFit.config  --gpu 0 --run_mode test \
              --checkpoint ../output/EntityQuestions/BaseModel/BitFit/ckpt.bin
