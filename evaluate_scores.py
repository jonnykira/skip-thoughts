import eval_classification
import openai.utils
import openai.encoder
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--restore_path', type=str, default=None,
                    help='Path to the trained model to evaluate from')
parser.add_argument('--task', type=str, default='MR',
                    help='1 of 4 classiicatin tasks. This can be MR, CR SUBJ or MPQA')
parser.add_argument('--data_path', type=str, default=None,
                    help='Path to the directory containing the data. Must match the data for the given task!')
# weights = openai.utils.extract_for_encoder(model_dir='/Users/jonathan/Desktop/Experiment/checkpoints/994')
#
# np.save('model.npy', weights)
args = parser.parse_args()

encoder = openai.encoder.Model(os.path.join(args.restore_path,'model.npy'))

# eval_nested_kfold(encoder, name, loc='./data/', k=10, seed=1234, use_nb=False)
scores = eval_classification.eval_nested_kfold(encoder,name=args.task,loc=args.data_path, use_nb=False)

# save the results vector in the results
np.save('./results/' + args.tasks, scores)
