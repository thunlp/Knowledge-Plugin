# Knowledge Plugin

Source codes and datasets for *Plug-and-Play Knowledge Injection for Pre-trained Language Models*.

We will update all datasets and checkpoints in this week.

## Preliminary

### Prepare Datasets

```bash
# TODO: datasets/{FewRel,Wiki80,WikiET,wiki20m,EntityQuestions}

# For EntityQuestions
cd datasets/EntityQuestions
wget https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip
unzip dataset.zip
mv dataset/train train
mv dataset/dev dev
mv dataset/test test
```

### Prepare Knowledge Embedding

```bash
# TODO: knowledge_embedding
```

## Train Downstream (Base) Models

We use $\text{BERT}_\text{base}$ as the backbone PLM in the experiments, and we consider four training methods for the adaptation on downstream tasks. We first train the downstream (base) models on four datasets using the four training methods. Note that `BERT` in the class/file name refers to the vanilla full-model fine-tuning.

```bash
cd code
bash run_FewRel_BaseModel.sh
bash run_Wiki80_BaseModel.sh
bash run_WikiET_BaseModel.sh
bash run_EntityQuestions_BaseModel.sh
```

In the following plug-and-play stages, the mapping networks are plugged to these downstream (base) models. You can download our checkpoints.

```bash
# TODO: {FewRel, Wiki80, WikiET, EntityQuestions}/BaseModel
```

## Map-Tuning on [Wikipedia Corpus] / [Downstream Data]

We freeze the PLM and train the mapping network by Mention-Masked Language Modeling (MMLM), which is called general map-tuning in the paper. After general map-tuning, the mapping network we get can be used in general plug-and-play injection, or we can fine-tune the PLM with the mapping network on downstream tasks.

We use the wikipedia corpus as the training data for general map-tuning with varying dropout probabilities (Empirically, 0.25 is a good choice.).

```bash
python MapTuning_Wikipedia.py --config ../mapping_networks/Wikipedia/NoDropout/default.config --gpu 0

python MapTuning_Wikipedia.py --config ../mapping_networks/Wikipedia/Dropout15/default.config --gpu 0
python MapTuning_Wikipedia.py --config ../mapping_networks/Wikipedia/Dropout25/default.config --gpu 0
python MapTuning_Wikipedia.py --config ../mapping_networks/Wikipedia/Dropout35/default.config --gpu 0
python MapTuning_Wikipedia.py --config ../mapping_networks/Wikipedia/Dropout45/default.config --gpu 0
```

We also train mapping networks on the downstream data. Note that these mapping networks are NOT used in plug-and-play injection.

```bash
python MapTuning_FewRel.py          --config ../mapping_networks/Downstream/FewRel/default.config          --gpu 0
python MapTuning_Wiki80.py          --config ../mapping_networks/Downstream/Wiki80/default.config          --gpu 0
python MapTuning_WikiET.py          --config ../mapping_networks/Downstream/WikiET/default.config          --gpu 0
python MapTuning_EntityQuestions.py --config ../mapping_networks/Downstream/EntityQuestions/default.config --gpu 0
```

You can also download our checkpoints.

```bash
# TODO: mapping_networks
```

## General Plug-and-Play Injection

After general map-tuning (on wikipedia corpus), the mapping network we get can be used in general plug-and-play injection. We directly plug the mapping network into each downstream model without additional training.

```bash
bash run_FewRel_Plug_General.sh
bash run_Wiki80_Plug_General.sh
bash run_WikiET_Plug_General.sh
bash run_EntityQuestions_Plug_General.sh
```

## Task-Specific Plug-and-Play Injection

```bash
bash run_FewRel_Plug_TaskSpecific.sh
python run.py --config config/Wiki80/Plug_TaskSpecific/default.config --run_mode map-tuning --checkpoint ../output/Wiki80/BaseModel/BERT/ckpt.bin --gpu 0
python run.py --config config/WikiET/Plug_TaskSpecific/default.config --run_mode map-tuning --checkpoint ../output/WikiET/BaseModel/BERT/ckpt.bin --gpu 0
python run.py --config config/EntityQuestions/Plug_TaskSpecific/default.config --run_mode map-tuning --checkpoint ../output/EntityQuestions/BaseModel/BERT/ckpt.bin --gpu 0
```

## Fine-Tuning with Mapping Network

We freeze the parameters of the mapping network and fine-tune the PLM on downstream tasks, during which we augment model inputs with mapped knowledge representations.

The mapping networks are trained on wikipedia corpus (`Wikipedia`) or on corresponding downstream data (`Downstream`).

```bash
bash run_FewRel_FineTuning.sh
bash run_Wiki80_FineTuning.sh
bash run_WikiET_FineTuning.sh
bash run_EntityQuestions_FineTuning.sh
```

You can download our checkpoints.

```bash
# TODO: {FewRel, Wiki80, WikiET, EntityQuestions}/FineTuning
```

## FewRel submission

As the test set (`datasets/FewRel/test_wiki.json`) of FewRel is not publicly released, you may have to contact the [authors](https://github.com/thunlp/FewRel) for the test set if you want to run our code on the test set by yourself. We recommende submitting the results to the [official leaderboard](https://codalab.lisn.upsaclay.fr/competitions/7395). The input data is [here](https://worksheets.codalab.org/worksheets/0x224557d3a319469c82b0eb2550a2219e) and you can download the data to `datasets/FewRel/submission`. You can get the submission file by `FewRel_submission.py` and here is one example command.

```bash
python FewRel_submission.py --config config/FewRel/Plug_General/BERT_5way1shot.config --gpu 0 --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin --data_path ../datasets/FewRel/submission --data_name test_wiki_input
```

We put the submission file of general plug-and-play injection (to BERT) in `output/FewRel/Plug_General/BERT_5way1shot`.

## Cite

If you use the code, please cite this paper:

```
@inproceedings{zhang2023plug,
  title={Plug-and-Play Knowledge Injection for Pre-trained Language Models},
  author={Zhang, Zhengyan and Zeng, Zhiyuan and Lin, Yankai and Wang, Huadong and Ye, Deming and Xiao, Chaojun and Han, Xu and Liu, Zhiyuan and Li, Peng and Sun, Maosong and Zhou, Jie},
  booktitle={Proceedings of ACL},
  year={2023}
}
```
