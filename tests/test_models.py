"""Forward-pass checks for every architecture, plus the results metrics.

Pretrained weight downloads are disabled so the suite runs offline.
"""

import functools

import pytest
import torch
import torchvision

import models as M
from metrics import RESULT_COLUMNS, classification_metrics


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    for name in ("vgg16", "mobilenet_v2", "resnet18"):
        original = getattr(torchvision.models, name)

        def offline(*args, _original=original, **kwargs):
            kwargs.pop("weights", None)
            kwargs.pop("pretrained", None)
            return _original(*args, weights=None, **kwargs)

        monkeypatch.setattr(M.models, name, functools.wraps(original)(offline))


def _batch(size):
    return torch.randn(2, 3, size, size), torch.randn(2, 3, size, size)


SINGLE_STREAM = [("VGG16", 224), ("Mobilenetv2", 224), ("Resnet18", 224)]
TWO_INPUT = [
    ("Flame_one_stream", 254, ["rgb", "ir", "both"]),
    ("Flame_two_stream", 254, ["both"]),
    ("Vgg_two_stream", 224, ["both"]),
    ("Mobilenetv2_two_stream", 224, ["both"]),
    ("Resnet18_two_stream", 224, ["both"]),
    ("Logistic_two_stream", 254, ["rgb", "ir", "both"]),
    ("LeNet5_one_stream", 254, ["rgb", "ir", "both"]),
    ("LeNet5_two_stream", 254, ["both"]),
]


@pytest.mark.parametrize("name,size", SINGLE_STREAM)
def test_single_stream_models_output_three_classes(name, size):
    net = getattr(M, name)(3).eval()
    rgb, _ = _batch(size)
    with torch.no_grad():
        assert net(rgb).shape == (2, 3)


@pytest.mark.parametrize(
    "name,size,mode",
    [(n, s, mode) for n, s, modes in TWO_INPUT for mode in modes],
)
def test_two_input_models_output_three_classes(name, size, mode):
    net = getattr(M, name)().eval()
    rgb, ir = _batch(size)
    with torch.no_grad():
        assert net(rgb, ir, mode=mode).shape == (2, 3)


def test_pretrained_backbones_request_default_weights(monkeypatch):
    seen = {}
    real_resnet18 = torchvision.models.resnet.resnet18  # unpatched

    def spy(**kwargs):
        seen.update(kwargs)
        return real_resnet18(weights=None)

    monkeypatch.setattr(M.models, "resnet18", spy)
    M.Resnet18(3)
    assert seen == {"weights": "DEFAULT"}


def test_classification_metrics_perfect_and_partial():
    target = torch.tensor([0, 1, 2, 0, 1, 2])
    perfect = classification_metrics(target, target)
    assert all(v == pytest.approx(1.0) for v in perfect.values())

    preds = torch.tensor([0, 2, 1, 0, 0, 1])
    m = classification_metrics(preds, target)
    # Micro F1 equals accuracy for single-label multiclass: 2 of 6 correct.
    assert m["micro_f1"] == pytest.approx(2 / 6)
    assert set(m) == set(RESULT_COLUMNS[3:])


def test_classification_metrics_accepts_float_tensors():
    # train.py accumulates labels with torch.cat into float tensors.
    target = torch.tensor([0.0, 1.0, 2.0])
    assert classification_metrics(target, target)["macro_f1"] == pytest.approx(1.0)


def test_train_script_imports():
    # Guards against dependency API breaks, like torchmetrics dropping
    # precision_recall, which stopped train.py from starting at all.
    import train  # noqa: F401
