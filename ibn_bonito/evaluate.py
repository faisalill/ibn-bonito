import time
import torch
import numpy as np
from itertools import starmap
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from bonito.data import load_numpy, load_script
from bonito.util import accuracy, poa, decode_ref, half_supported
from bonito.util import init, load_model, permute

from torch.utils.data import DataLoader


poas = []
print("* loading data")
_, valid_loader_kwargs = load_numpy(None, args.directory)

dataloader = DataLoader(
    batch_size=args.batchsize, num_workers=4, pin_memory=True, **valid_loader_kwargs
)

accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=args.min_coverage)

for w in [int(i) for i in args.weights.split(",")]:
    seqs = []

    print("* loading model", w)
    model = load_model(args.model_directory, args.device, weights=w)
    mean = model.config.get("standardisation", {}).get("mean", 0.0)
    stdev = model.config.get("standardisation", {}).get("stdev", 1.0)
    print(f"* * signal standardisation params: mean={mean}, stdev={stdev}")

    print("* calling")
    t0 = time.perf_counter()

    targets = []

    with torch.no_grad():
        for data, target, *_ in dataloader:
            data = (data - mean) / stdev

            targets.extend(torch.unbind(target, 0))
            if half_supported():
                data = data.type(torch.float16).to(args.device)
            else:
                data = data.to(args.device)

            log_probs = model(data)

            if hasattr(model, "decode_batch"):
                seqs.extend(model.decode_batch(log_probs))
            else:
                seqs.extend([model.decode(p) for p in permute(log_probs, "TNC", "NTC")])

    duration = time.perf_counter() - t0

    refs = [decode_ref(target, model.alphabet) for target in targets]
    accuracies = [
        accuracy_with_cov(ref, seq) if len(seq) else 0.0 for ref, seq in zip(refs, seqs)
    ]

    if args.poa:
        poas.append(sequences)

    print("* mean      %.2f%%" % np.mean(accuracies))
    print("* median    %.2f%%" % np.median(accuracies))
    print("* time      %.2f" % duration)
    print("* samples/s %.2E" % (args.chunks * data.shape[2] / duration))

if args.poa:
    print("* doing poa")
    t0 = time.perf_counter()
    # group each sequence prediction per model together
    poas = [list(seq) for seq in zip(*poas)]
    consensuses = poa(poas)
    duration = time.perf_counter() - t0
    accuracies = list(
        starmap(accuracy_with_coverage_filter, zip(references, consensuses))
    )

    print("* mean      %.2f%%" % np.mean(accuracies))
    print("* median    %.2f%%" % np.median(accuracies))
    print("* time      %.2f" % duration)
