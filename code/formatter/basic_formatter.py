from mapper import AffineMapper
from all_embedding import KE_Embedding, get_PLMembeddings

class basic_formatter() :
    def __init__(self, config) :
        self.config = config
        self.max_length = config.getint("data", "max_seq_length")
        mapper_type = config.get("mapper", "mapper_type")
        if mapper_type != "None" :
            self.keembedding = KE_Embedding(config)
            mapper_path = config.get("mapper", "mapper_path")
            if mapper_type == "Affine" :
                self.mapper = AffineMapper(config.getint("mapper", "input_dim"), config.getint("mapper", "output_dim"))
                self.mapper.load(mapper_path)
            else :
                print("There is no mapper_type called {}".format(mapper_type))
                assert(False)
        else :
            self.mapper = None
    def get_mapper(self) :
        return self.mapper

class PLM_formatter(basic_formatter) :
    def __init__(self, config, BaseModel, BaseTokenizer) :
        super().__init__(config)
        self.PLM_embedding = get_PLMembeddings(BaseModel.from_pretrained(config.get("model", "PLM_path")))
        self.NeedMapper = [False] * len(self.PLM_embedding)
        self.tokenizer = BaseTokenizer.from_pretrained(config.get("model", "PLM_path"))
    def get_embedding(self) :
        return self.PLM_embedding
    def get_NeedMapper(self) :
        return self.NeedMapper
    def text_to_input(self, text : str, need_padding = True) :
        text = text.split()
        token = []
        for substring in text :
            token += self.tokenizer.tokenize(substring)
            if len(token) > self.max_length - 2 :
                break
        if len(token) > self.max_length - 2 :
            token = token[: (self.max_length - 2)]
        token = ["[CLS]"] + token + ["[SEP]"]
        token = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * len(token)
        if need_padding :
            padding = [0] * (self.max_length - len(token))
            token += padding
            mask += padding
        return token, mask
    #unmapped
    def insert_Qid(self, word, Qid : str) :
        if not word in self.tokenizer.vocab :
            self.tokenizer.vocab[word] = len(self.tokenizer.vocab)
            self.tokenizer.ids_to_tokens[self.tokenizer.vocab[word]] = word
            emb, need = self.keembedding.query_Qid(Qid)
            self.PLM_embedding.append(emb)
            self.NeedMapper.append(need)
    def insert_Cid(self, word, Cid : str) :
        if not word in self.tokenizer.vocab :
            self.tokenizer.vocab[word] = len(self.tokenizer.vocab)
            self.tokenizer.ids_to_tokens[self.tokenizer.vocab[word]] = word
            emb, need = self.keembedding.query_Cid(Cid)
            self.PLM_embedding.append(emb)
            self.NeedMapper.append(need)
    #unmapped
    def insert_id(self, word, id : int) :
        if not word in self.tokenizer.vocab :
            self.tokenizer.vocab[word] = len(self.tokenizer.vocab)
            self.tokenizer.ids_to_tokens[self.tokenizer.vocab[word]] = word
            emb, need = self.keembedding.query_id(id)
            self.PLM_embedding.append(emb)
            self.NeedMapper.append(need)

from transformers import BertModel, BertTokenizer
class bert_formatter(PLM_formatter) :
    def __init__(self, config) :
        super().__init__(config, BertModel, BertTokenizer)