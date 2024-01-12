# Knowledge Plugin (sub-project of OpenSKL)

Source codes and datasets for *[Plug-and-Play Knowledge Injection for Pre-trained Language Models](https://arxiv.org/abs/2305.17691)* (ACL 2023).

## Preliminary

### Prepare Datasets

```bash
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/PlugDataset.tar
tar -xvf PlugDataset.tar
rm -r PlugDataset.tar

# For the training and dev set of FewRel
cd datasets/FewRel
wget https://raw.githubusercontent.com/thunlp/FewRel/master/data/train_wiki.json
wget https://raw.githubusercontent.com/thunlp/FewRel/master/data/val_wiki.json
cd ../..

# For EntityQuestions
mkdir datasets/EntityQuestions
cd datasets/EntityQuestions
wget https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip
unzip dataset.zip
mv dataset/train train
mv dataset/dev dev
mv dataset/test test
rm -r dataset.zip
rm -rf dataset
wget https://raw.githubusercontent.com/princeton-nlp/EntityQuestions/master/relation_query_templates.json
cd ../..
```

### Prepare Knowledge Embedding

```bash
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/knowledge_embedding.tar
tar -xvf knowledge_embedding.tar
rm -r knowledge_embedding.tar
mkdir knowledge_embedding
mv wikipedia knowledge_embedding/wikipedia
```

### Checkpoints

You can download all the checkpoints of models trained by us.

```bash
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/PlugAndPlay_ckpt.tar
tar -xvf PlugAndPlay_ckpt.tar
rm -r PlugAndPlay_ckpt.tar
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

In the following plug-and-play stages, the mapping networks are plugged to these downstream (base) models. You can download our checkpoints. If you have already downloaded our checkpoints, then the checkpoints of downstream (base) models are in `output/{FewRel, Wiki80, WikiET, EntityQuestions}/BaseModel`.

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

We also train mapping networks on the downstream data. Note that these mapping networks are NOT used in plug-and-play injection and they are used in the experiments of fine-tuning with mapping networks.

```bash
python MapTuning_FewRel.py          --config ../mapping_networks/Downstream/FewRel/default.config          --gpu 0
python MapTuning_Wiki80.py          --config ../mapping_networks/Downstream/Wiki80/default.config          --gpu 0
python MapTuning_WikiET.py          --config ../mapping_networks/Downstream/WikiET/default.config          --gpu 0
python MapTuning_EntityQuestions.py --config ../mapping_networks/Downstream/EntityQuestions/default.config --gpu 0
```

You can also download our checkpoints of mapping networks.

```bash
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/mapping_networks.tar
tar -xvf mapping_networks.tar
rm -r mapping_networks.tar
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

If you have already downloaded our checkpoints, then the checkpoints of models finetuned with mapping networks are in `output/{FewRel, Wiki80, WikiET, EntityQuestions}/FineTuning`.

## FewRel Submission

We recommende submitting the results to the [official leaderboard](https://codalab.lisn.upsaclay.fr/competitions/7395). The input data is [here](https://worksheets.codalab.org/worksheets/0x224557d3a319469c82b0eb2550a2219e) and we downloaded the data to `datasets/FewRel/submission`. You can get the submission file by `FewRel_submission.py` and here is one example command.

```bash
python FewRel_submission.py --config config/FewRel/Plug_General/BERT_5way1shot.config --gpu 0 --checkpoint ../output/FewRel/BaseModel/BERT_5way1shot/ckpt.bin --data_path ../datasets/FewRel/submission --data_name test_wiki_input
```

We put the submission file of general plug-and-play injection (to BERT) in `output/FewRel/Plug_General/BERT_5way1shot`.

## TransR Version

To use TransR knowledge embeddings, please first download the embeddings:

```bash
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/transr.npy
mv transr.npy knowledge_embedding/wikipedia/transr.npy
```

Then, in config files, change `ke_path = ../knowledge_embedding/wikipedia/transe.npy` to `ke_path = ../knowledge_embedding/wikipedia/transr.npy`. We also need to change `input_dim = 128` to `input_dim = 228`. All the running commands are the same as before.

We also provide our checkpoints of mapping networks for TransR knowledge embeddings:

```bash
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/mapping_networks_TransR.tar
tar -xvf mapping_networks_TransR.tar
rm -r mapping_networks_TransR.tar
```

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

******************
## About OpenSKL
OpenSKL project aims to harness the power of both structured knowledge and natural languages via representation learning. All sub-projects of OpenSKL, under the categories of **Algorithm**, **Resource** and **Application**, are as follows.

- **Algorithm**: 
  - [OpenKE](https://www.github.com/thunlp/OpenKE)
    - An effective and efficient toolkit for representing structured knowledge in large-scale knowledge graphs as embeddings, with <a href="https://ojs.aaai.org/index.php/AAAI/article/view/9491/9350"> TransR</a> and  <a href="https://aclanthology.org/D15-1082.pdf">PTransE</a> as key features to handle complex relations and relational paths.
    - This toolkit also includes three repositories:
       - [KB2E](https://www.github.com/thunlp/KB2E)
       - [TensorFlow-Transx](https://www.github.com/thunlp/TensorFlow-Transx)
       - [Fast-TransX](https://www.github.com/thunlp/Fast-TransX)
  - [ERNIE](https://github.com/thunlp/ERNIE)
    - An effective and efficient toolkit for augmenting pre-trained language models with knowledge graph representations.
  - [OpenNE](https://www.github.com/thunlp/OpenNE)
    - An effective and efficient toolkit for representing nodes in large-scale graphs as embeddings, with [TADW](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) as key features to incorporate text attributes of nodes.
  - [OpenNRE](https://www.github.com/thunlp/OpenNRE)
    - An effective and efficient toolkit for implementing neural networks for extracting structured knowledge from text, with [ATT](https://aclanthology.org/P16-1200.pdf) as key features to consider relation-associated text information.
    - This toolkit also includes two repositories:
      - [JointNRE](https://www.github.com/thunlp/JointNRE)
      - [NRE](https://github.com/thunlp/NRE)
- **Resource**:
  - The embeddings of large-scale knowledge graphs pre-trained by OpenKE, covering three typical large-scale knowledge graphs: Wikidata, Freebase, and XLORE. The embeddings are free to use under the [MIT license](https://opensource.org/license/mit/), and please click the following link to submit [download requests](http://139.129.163.161/download/wikidata).
  - OpenKE-Wikidata
    - Wikidata is a free and collaborative database, collecting structured data to provide support for Wikipedia. The original Wikidata contains 20,982,733 entities, 594 relations and 68,904,773 triplets. In particular, Wikidata-5M is the core subgraph of Wikidata, containing  5,040,986 high-frequency entities from Wikidata with their corresponding 927 relations and 24,267,796 triplets.
    - [TransE version](http://139.129.163.161/download/wikidata): Knowledge embeddings of Wikidata pre-trained by OpenKE. 
    - [TransR version](http://139.129.163.161/download/wikidata) of Wikidata-5M: Knowledge embeddings of Wikidata-5M pre-trained by OpenKE.
  - OpenKE-Freebase
    - Freebase was a large collaborative knowledge base consisting of data composed mainly by its community members. It was an online collection of structured data harvested from many sources. Freebase contains 86,054,151 entities, 14,824 relations and 338,586,276 triplets.
    - [TransE version](http://139.129.163.161/download/wikidata): Knowledge embeddings of Freebase pre-trained by OpenKE. 
  - OpenKE-XLORE
    - XLORE is one of the most popular Chinese knowledge graphs developed by THUKEG. XLORE contains 10,572,209 entities, 138,581 relations and 35,954,249 triplets.
    - [TransE version](http://139.129.163.161/download/wikidata): Knowledge embeddings of XLORE pre-trained by OpenKE.
- **Application**:   
    - [Knowledge-Plugin](https://github.com/THUNLP/Knowledge-Plugin)
      - An effective and efficient toolkit of plug-and-play knowledge injection for pre-trained language models. Knowledge-Plugin is general for all kinds of knowledge graph embeddings mentioned above. In the toolkit, we plug the TransR version of Wikidata-5M into BERT as an example of applications. With the TransR embedding, we enhance the knowledge ability of BERT without fine-tuning the original model, e.g., up to 8% improvement on question answering.
