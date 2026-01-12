from transformers import BertTokenizer, BertModel
import torch.nn as nn

class DPRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.p_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.p_cls_list = []
        self.q_cls_list = []
        
    def load_checkpoint(self, epoch_dir):
        self.q_encoder = BertModel.from_pretrained(f"{epoch_dir}/q_encoder")
        self.p_encoder = BertModel.from_pretrained(f"{epoch_dir}/p_encoder")
        self.tokenizer = BertTokenizer.from_pretrained(f"{epoch_dir}/tokenizer")
        
    # 5 cls tokens [CLS] chunk1 [CLS] chunk2 ... [CLS] chunk5 [SEP]
    # chucks: 25, 25, 24, 24, 24
    def tokenize(self, text_a, text_b, MAX_LENGTH):
        if text_b == None:
            tokens = self.tokenizer( # tokenizer for questions
                text_a,
                max_length=MAX_LENGTH,
                return_tensors="pt",
                padding=False,
                truncation=False
            )
        else:
            tokens = self.tokenizer( # tokenizer for passages
                text_a, text_b,
                max_length=MAX_LENGTH,
                return_tensors="pt",
                padding=False,
                truncation=False
            )
        return tokens
        
    def encode_questions(self, q_input):
        cls_tokens = self.q_encoder(**q_input).last_hidden_state[:, 0, :]
        self.q_cls_list.append(cls_tokens)
        return cls_tokens

    def encode_passages(self, p_input):
        cls_tokens = self.p_encoder(**p_input).last_hidden_state[:, 0, :]
        self.p_cls_list.append(cls_tokens)
        return cls_tokens
    
    def get_modified_cls(self, type_val):
        if type_val == "passage":
            return self.p_cls_list
        if type_val == "question":
            return self.q_cls_list

    def forward(self, q_emb, p_emb):
        return q_emb @ p_emb.T
    
if __name__ == "__main__":
    model = DPRModel()
    print(model.tokenize("adsf asdf b d d kekekf", None, 10))