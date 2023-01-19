from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np


def sapbert_predict(model_folder, all_names, use_gpu=True):
    # load sapbert
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    if use_gpu:
        model = AutoModel.from_pretrained(model_folder).cuda(0)
    else:
        model = AutoModel.from_pretrained(model_folder)

    bs = 128
    all_reps = []
    for i in tqdm(np.arange(0, len(all_names), bs)):
        toks = tokenizer.batch_encode_plus(all_names[i:i + bs],
                                           padding="max_length",
                                           max_length=25,
                                           truncation=True,
                                           return_tensors="pt")
        if use_gpu:
            toks_cuda = {}
            for k,v in toks.items():
               toks_cuda[k] = v.cuda(0)
            output = model(**toks_cuda)
        else:
            output = model(**toks)
        cls_rep = output[0][:, 0, :]

        all_reps.append(cls_rep.cpu().detach().numpy())
    all_reps_emb = np.concatenate(all_reps, axis=0)
    return model, tokenizer, all_reps_emb
