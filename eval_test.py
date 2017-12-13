import eval_classification
import openai.utils
import openai.encoder


encoder = openai.encoder.Model()

# eval_nested_kfold(encoder, name, loc='./data/', k=10, seed=1234, use_nb=False)

scores = eval_classification.eval_nested_kfold(encoder,name='CR',loc='/Users/jonathan/Downloads/data/customerr', use_nb=False)
