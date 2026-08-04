"""Causal inference across reference frames in a feedforward network.

    cmsi.data       generative model, analytical observer, population encoders
    cmsi.models     the network, its loss, the training loop
    cmsi.analysis   comparisons against the observer
    cmsi.viz        figures, grouped as inputs / training / model results
    cmsi.utils      config, paths, seeding, file IO

Typical use:

    from cmsi.utils import load_config, run_dir
    from cmsi.data import make_dataset, subset
    from cmsi.models import train, predict
    from cmsi import analysis

    cfg = load_config()
    d = make_dataset(cfg)
    model, history, splits = train(d, cfg)
    test = subset(d, splits["test"])
    pred = predict(model, test["X"])
    analysis.print_accuracy(analysis.accuracy(pred, test["Y"], d["target_names"]))
"""

__version__ = "0.1.0"
