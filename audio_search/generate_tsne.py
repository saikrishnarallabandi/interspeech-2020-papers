import torch
from sklearn.manifold import TSNE
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from utils import *
import nocond as nc
import vqvae as vqvae
import wavernn1 as wr
import env as env
import argparse
import re
import config

parser = argparse.ArgumentParser(description='Train or run some neural net')
parser.add_argument('--generate', '-g', action='store_true')
parser.add_argument('--float', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--load', '-l')
parser.add_argument('--scratch', action='store_true')
parser.add_argument('--model', '-m')
parser.add_argument('--force', action='store_true', help='skip the version check')
parser.add_argument('--count', '-c', type=int, default=3, help='size of the test set')
parser.add_argument('--partial', action='append', default=[], help='model to partially load')
parser.add_argument('--path', '-p')
args = parser.parse_args()

if args.float and args.half:
    sys.exit('--float and --half cannot be specified together')

if args.float:
    use_half = False
elif args.half:
    use_half = True
else:
    use_half = False

model_type = args.model or 'vqvae'

model_name = "vqvae_multispeaker_baseline"

if model_type == 'vqvae':
    model_fn = lambda dataset: vqvae.Model(rnn_dims=896, fc_dims=896,
                  upsample_factors=(4, 4, 4), normalize_vq=True, noise_x=True, noise_y=True).cuda()
    dataset_type = 'unknown'
elif model_type == 'wavernn':
    model_fn = lambda dataset: wr.Model(rnn_dims=896, fc_dims=896, pad=2,
                  upsample_factors=(4, 4, 4), feat_dims=80).cuda()
    dataset_type = 'single'
elif model_type == 'nc':
    model_fn = lambda dataset: nc.Model(rnn_dims=896, fc_dims=896).cuda()
    dataset_type = 'single'
else:
    sys.exit(f'Unknown model: {model_type}')

if dataset_type == 'unknown':
    data_path = config.unknown_speaker_data_path
    dataset = env.UnknownSpeakerDataset(data_path)
    total = len(dataset)
    train_i = int(0.9 * total)
    train_index = [i for i in range(0, train_i)]
    test_index = [i for i in range(train_i, total)]
else:
    raise RuntimeError('bad dataset type')

print(f'dataset size: {len(dataset)}')

model = model_fn(dataset)

if use_half:
    model = model.half()

for partial_path in args.partial:
    model.load_state_dict(torch.load(partial_path), strict=False)

paths = env.Paths(model_name, data_path)

if args.scratch or args.load == None and not os.path.exists(paths.model_path()):
    # Start from scratch
    step = 0
else:
    if args.load:
        prev_model_name = re.sub(r'_[0-9]+$', '', re.sub(r'\.pyt$', '', os.path.basename(args.load)))
        prev_model_basename = prev_model_name.split('_')[0]
        model_basename = model_name.split('_')[0]
        if prev_model_basename != model_basename and not args.force:
            sys.exit(
                f'refusing to load {args.load} because its basename ({prev_model_basename}) is not {model_basename}')
        if args.generate:
            paths = env.Paths(prev_model_name, data_path)
        prev_path = args.load
    else:
        prev_path = paths.model_path()
    step = env.restore(prev_path, model)

print("Computing TSNE")
path = args.path + "embedding_audio_discrete.png"
# search_discretes = [x for x in sorted(os.listdir(path + '/audio_discrete/'))]
# for search_discrete_file in
# phone_embedding = np.load(f'{path}/audio_discrete/{search_file_name}.npy')
phone_embedding = model.vq.embedding0
phone_embedding = list(phone_embedding[0].cpu().detach().numpy())
phone_embedded = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(phone_embedding)
# model.num_classes
num_classes = 256

y = phone_embedded[:, 0]
z = phone_embedded[:, 1]

fig, ax = plt.subplots()
ax.scatter(y, z)

for i in range(num_classes):
    ax.annotate(i, (y[i], z[i]))
plt.tight_layout()
plt.savefig(path, format="png")
plt.close()
