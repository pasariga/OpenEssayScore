"""Text-only features shared by training and standalone offline inference."""
import re
import time
import numpy as np

FEATURE_NAMES = ['log_chars', 'log_cjk', 'log_paragraphs', 'log_sentences',
                 'unique_character_ratio', 'punctuation_ratio', 'log_mean_sentence',
                 'digit_ratio', 'latin_ratio', 'dialogue_mark_ratio']


def structural(texts):
    rows=[]
    for text in texts:
        n=max(1,len(text))
        sentences=[s for s in re.split(r'[。！？!?]+',text) if s.strip()]
        paragraphs=[p for p in text.splitlines() if p.strip()]
        rows.append([np.log1p(n),np.log1p(len(re.findall(r'[\u4e00-\u9fff]',text))),
                     np.log1p(len(paragraphs)),np.log1p(len(sentences)),len(set(text))/n,
                     len(re.findall(r'[，。！？、；：,.!?;:]',text))/n,
                     np.log1p(n/max(1,len(sentences))),len(re.findall(r'\d',text))/n,
                     len(re.findall(r'[a-zA-Z]',text))/n,text.count('“')/n])
    return np.asarray(rows,dtype=np.float32)


class Encoder:
    def __init__(self,path,device='cpu',batch_size=16):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch=torch
        self.device=device
        self.batch_size=batch_size
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
        self.model=AutoModel.from_pretrained(str(path),local_files_only=True,
                                             trust_remote_code=False,use_safetensors=True,
                                             attn_implementation='eager').to(device).eval()
        self.parameter_count=sum(p.numel() for p in self.model.parameters())

    def encode(self,texts):
        # Cover full essays: nonoverlapping 510-token chunks + special tokens.
        # No head-only truncation. Mean normalized CLS across all chunks.
        torch=self.torch
        chunks,owners=[],[]
        counts=[]
        for index,text in enumerate(texts):
            ids=self.tokenizer.encode(text,add_special_tokens=False,truncation=False)
            start_count=len(chunks)
            for start in range(0,max(1,len(ids)),510):
                tokens=self.tokenizer.build_inputs_with_special_tokens(ids[start:start+510])
                chunks.append({'input_ids':tokens,'attention_mask':[1]*len(tokens)})
                owners.append(index)
            counts.append(len(chunks)-start_count)
        out=np.zeros((len(texts),self.model.config.hidden_size),dtype=np.float32)
        if self.device=='cuda': torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            for i in range(0,len(chunks),self.batch_size):
                batch=self.tokenizer.pad(chunks[i:i+self.batch_size],padding=True,return_tensors='pt')
                batch={k:v.to(self.device) for k,v in batch.items()}
                if self.device=='cuda':
                    with torch.autocast(device_type='cuda',dtype=torch.float16):
                        vectors=self.model(**batch).last_hidden_state[:,0]
                else:
                    vectors=self.model(**batch).last_hidden_state[:,0]
                vectors=torch.nn.functional.normalize(vectors.float(),p=2,dim=1).cpu().numpy()
                for j,vector in enumerate(vectors): out[owners[i+j]]+=vector
        out/=np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-12)
        if self.device=='cuda': torch.cuda.synchronize()
        return out,{'seconds':time.perf_counter()-start,'essays':len(texts),
                    'chunks':len(chunks),'max_chunks_per_essay':max(counts),'device':self.device}
