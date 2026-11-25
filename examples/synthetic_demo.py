"""A CPU-only training example using invented text and arbitrary teaching labels."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

TRAINING = [
    ("操场下雨了。", 3.0),
    ("下午我去图书馆借书。", 4.0),
    ("我把纸船放进水沟。风大起来，纸船贴着石头打了个转。", 7.0),
    ("雨停后我走到操场边，看见水洼里倒映着跑道上的白线。", 7.5),
    ("我在图书馆找地图，翻到河流那一页，又用铅笔记下回家后要查的问题。", 8.0),
    ("我和同伴扶住倒伏的花苗，用小石头垫牢，再把多余的水从花盆边倒掉。", 8.5),
]
SAMPLE = "放学后风停了，我在操场看见纸船卡在石头旁，便用一根树枝把它推回水面。"


def main():
    texts, labels = zip(*TRAINING)
    pipeline = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 2)),
        Ridge(alpha=1.0),
    )
    pipeline.fit(texts, labels)
    [prediction] = pipeline.predict([SAMPLE])
    print(f"Synthetic workflow only | demo output: {max(0.0, min(10.0, prediction)):.2f}/10")


if __name__ == "__main__":
    main()
