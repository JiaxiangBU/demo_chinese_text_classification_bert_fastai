# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/wshuyi/demo_chinese_text_classification_bert_fastai/blob/master/demo_refactored_dianping_classification_with_BERT_fastai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + colab={} colab_type="code" id="T-HkYxo2GZT2"
from fastai.text import *

# + colab={} colab_type="code" id="DUU3lFXrXAUw"
# !wget https://github.com/wshuyi/public_datasets/raw/master/dianping.csv

# + colab={} colab_type="code" id="nDfYplnJM0A-"
df = pd.read_csv("dianping.csv")

# + colab={} colab_type="code" id="Z8m0iT14NAJz"
from sklearn.model_selection import train_test_split

# + colab={} colab_type="code" id="yUWWbsurNHRh"
train, test = train_test_split(df, test_size=.2, random_state=2)

# + colab={} colab_type="code" id="YTlBj0_vNOGs"
train, valid = train_test_split(train, test_size=.2, random_state=2)

# + colab={} colab_type="code" id="5M58ohRXNRma"
len(train)

# + colab={} colab_type="code" id="gwYryualNV1U"
len(valid)

# + colab={} colab_type="code" id="iWWYv75WNW9l"
len(test)

# + colab={} colab_type="code" id="4ot3umGoNSw2"
train.head()

# + colab={} colab_type="code" id="Q3gQIfrZJ8h7"
# !pip install pytorch-transformers

# + colab={} colab_type="code" id="DGPWStCvKASr"
from pytorch_transformers import BertTokenizer, BertForSequenceClassification

# + colab={} colab_type="code" id="3dS1_jNESiEv"
bert_model = "bert-base-chinese"
max_seq_len = 128
batch_size = 32

# + colab={} colab_type="code" id="JiKPDWViStZS"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

# + colab={} colab_type="code" id="LxXk9yB1V0nB"
list(bert_tokenizer.vocab.items())[2000:2005]

# + colab={} colab_type="code" id="NYIrjtkHW2F2"
bert_vocab = Vocab(list(bert_tokenizer.vocab.keys()))


# + colab={} colab_type="code" id="aHXvwk8VGyq1"
class BertFastaiTokenizer(BaseTokenizer):
    def __init__(self, tokenizer, max_seq_len=128, **kwargs):
        self.pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t):
        return ["[CLS]"] + self.pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]


# + colab={} colab_type="code" id="BW8yrL48W7iX"
tok_func = BertFastaiTokenizer(bert_tokenizer, max_seq_len=max_seq_len)

# + colab={} colab_type="code" id="0HD1qCGCVMMe"
bert_fastai_tokenizer = Tokenizer(
    tok_func=tok_func,
    pre_rules = [],
    post_rules = []
)

# + colab={} colab_type="code" id="NTOtOYvvNfVh"
path = Path(".")

# + colab={} colab_type="code" id="2FmN5wWFNbcj"
databunch = TextClasDataBunch.from_df(path, train, valid, test,
                  tokenizer=bert_fastai_tokenizer,
                  vocab=bert_vocab,
                  include_bos=False,
                  include_eos=False,
                  text_cols="comment",
                  label_cols='sentiment',
                  bs=batch_size,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )

# + colab={} colab_type="code" id="lAH_8yNQOEI7"
databunch.show_batch()


# + colab={} colab_type="code" id="g5wzcUQ1YJ7y"
class MyNoTupleModel(BertForSequenceClassification):
  def forward(self, *args, **kwargs):
    return super().forward(*args, **kwargs)[0]


# + colab={} colab_type="code" id="8bWsk4psVJd2"
bert_pretrained_model = MyNoTupleModel.from_pretrained(bert_model, num_labels=2)

# + colab={} colab_type="code" id="NBM9tcDdVb8P"
loss_func = nn.CrossEntropyLoss()

# + colab={} colab_type="code" id="zoZZjQ9pPbdt"
learn = Learner(databunch, 
                bert_pretrained_model,
                loss_func=loss_func,
                metrics=accuracy)

# + colab={} colab_type="code" id="HU-sNOHfTpTp"
learn.lr_find()

# + colab={} colab_type="code" id="NPFIFxcnUB9P"
learn.recorder.plot()

# + colab={} colab_type="code" id="pDJDRZRsRGFI"
learn.fit_one_cycle(2, 2e-5)


# + colab={} colab_type="code" id="nHNMEpOAH3yN"
def dumb_series_prediction(n):
  preds = []
  for loc in range(n):
    preds.append(int(learn.predict(test.iloc[loc]['comment'])[1]))
  return preds


# + colab={} colab_type="code" id="NuhWUTdhn_UF"
preds = dumb_series_prediction(len(test))

# + colab={} colab_type="code" id="PvPRjZ_xoB80"
preds[:10]

# + colab={} colab_type="code" id="x9NCp8k5hcnd"
from sklearn.metrics import classification_report, confusion_matrix

# + colab={} colab_type="code" id="lcJbaF-7hezH"
print(classification_report(test.sentiment, preds))

# + colab={} colab_type="code" id="HWZXJ3iw-Zdv"
print(confusion_matrix(test.sentiment, preds))

# + colab={} colab_type="code" id="4IKLzdgDp2j8"

