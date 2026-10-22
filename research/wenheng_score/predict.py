"""Offline CPU scoring CLI. Does not call any third-party scoring API."""
import argparse
import json
from pathlib import Path
import re
import unicodedata
import joblib
import numpy as np
from features import Encoder, structural


def clean(text):
    text=unicodedata.normalize('NFKC',text).strip()
    text=re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}','[EMAIL]',text)
    return re.sub(r'(?<!\d)1[3-9]\d{9}(?!\d)','[PHONE]',text)


class Scorer:
    def __init__(self,model_dir,family=None,encoder_dir=None):
        self.path=Path(model_dir)
        metadata=json.loads((self.path/'model_metadata.json').read_text(encoding='utf-8'))
        self.family=family or metadata['selected_family']
        if self.family not in ['character_tfidf_ridge','length_ridge','semantic_ridge','semantic_structure_ridge']:
            raise ValueError('Unknown scorer family')
        file=self.path/f'{self.family}.joblib'
        self.model=joblib.load(file if file.exists() else self.path/'best_scorer.joblib')
        self.encoder=None
        if self.family=='character_tfidf_ridge':
            self.vectorizer=joblib.load(self.path/'lexical_vectorizer.joblib')
        elif self.family.startswith('semantic'):
            directory=Path(encoder_dir) if encoder_dir else self.path/'encoder'
            if not directory.exists():
                raise ValueError('Semantic scorer needs an explicit offline --encoder-dir')
            import torch
            torch.set_num_threads(2)
            self.encoder=Encoder(directory,device='cpu',batch_size=4)

    def predict(self,texts):
        texts=[clean(t) for t in texts]
        if any(len(re.findall(r'[\u4e00-\u9fff]',t))<20 for t in texts):
            raise ValueError('At least 20 Chinese characters per essay required for this research model')
        if self.family=='character_tfidf_ridge':x=self.vectorizer.transform(texts)
        elif self.family=='length_ridge':x=structural(texts)[:,:1]
        else:
            x,_=self.encoder.encode(texts)
            if self.family=='semantic_structure_ridge':x=np.concatenate([x,structural(texts)],axis=1)
        return np.clip(self.model.predict(x),0,10)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--encoder-dir',type=Path)
    parser.add_argument('--family')
    parser.add_argument('--essay-file',type=Path,required=True)
    args=parser.parse_args()
    scorer=Scorer(args.model_dir,args.family,args.encoder_dir)
    text=args.essay_file.read_text(encoding='utf-8-sig')
    score=float(scorer.predict([text])[0])
    print(json.dumps({'model':'WenHeng-Score v1 research baseline','family':scorer.family,
                      'score_0_10':score,'rounded_score_0_10':int(np.rint(score)),
                      'device':'cpu','offline':True,'status':'research_only',
                      'limitations':['primary-school benchmark only; not validated for all grades',
                                     'not a teacher replacement; does not evaluate prompt relevance',
                                     '0-10 benchmark scale; do not convert to school exam marks without calibration']},
                     ensure_ascii=False,indent=2))


if __name__=='__main__':main()
