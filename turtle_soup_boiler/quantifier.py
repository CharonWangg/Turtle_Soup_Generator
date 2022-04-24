from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer


class Quantifier:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    # TODO: add support for fast sentiment analysis
    def fast_quantify(self, sentence):
        raise NotImplementedError

    def quantify(self, sentence):
        scores = self.model(**self.tokenizer(sentence, return_tensors="pt"))[0][0].detach()
        scores = scores.softmax(dim=-1)
        return {"positive": scores[2].item(), "neutral": scores[1].item(), "negative": scores[0].item()}

    def get_sentiment_quantity(self, sentence):
        scores = self.quantify(sentence)
        # get the highest score in the dictionary
        score = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0]
        # map the score to a strength class
        return self.mapping_quantity_to_class(score)

    def mapping_quantity_to_class(self, score_dict):
        thr = {0: "extremely not", 1: "extremely not", 2: "not", 3: "a little not",
               4: "not very", 5: "little", 6: "slightly", 7: "", 8: "very", 9: "extremely"}
        # return {score_dict[0]: thr[int(score_dict[1]*10)]}
        return thr[int(score_dict[1] * 10)]





if __name__ == "__main__":
    quantifier = Quantifier()
    print(quantifier.get_sentiment_quantity("Good night 😊"))
