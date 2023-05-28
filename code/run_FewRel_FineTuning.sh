python run.py --config config/FewRel/FineTuning/Wikipedia_5way1shot.config  --gpu 0
python run.py --config config/FewRel/FineTuning/Downstream_5way1shot.config --gpu 0

python run.py --config config/FewRel/FineTuning/Wikipedia_5way5shot.config  --gpu 0  --run_mode test \
              --checkpoint ../output/FewRel/FineTuning/Wikipedia_5way1shot/ckpt.bin
python run.py --config config/FewRel/FineTuning/Wikipedia_10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/FineTuning/Wikipedia_5way1shot/ckpt.bin
python run.py --config config/FewRel/FineTuning/Wikipedia_10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/FineTuning/Wikipedia_5way1shot/ckpt.bin

python run.py --config config/FewRel/FineTuning/Downstream_5way5shot.config  --gpu 0  --run_mode test \
              --checkpoint ../output/FewRel/FineTuning/Downstream_5way1shot/ckpt.bin
python run.py --config config/FewRel/FineTuning/Downstream_10way1shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/FineTuning/Downstream_5way1shot/ckpt.bin
python run.py --config config/FewRel/FineTuning/Downstream_10way5shot.config --gpu 0 --run_mode test \
              --checkpoint ../output/FewRel/FineTuning/Downstream_5way1shot/ckpt.bin
