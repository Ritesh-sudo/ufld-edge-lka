import pytest
import torch

from ufld.device import get_device


def test_auto_matches_availability():
    expected = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    assert get_device().type == expected


def test_cpu_always_available():
    assert get_device("cpu").type == "cpu"


def test_unavailable_device_raises_instead_of_falling_back():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            get_device("cuda")


def test_tensor_runs_on_chosen_device():
    device = get_device()
    x = torch.ones(2, 2, device=device)
    assert (x @ x).tolist() == [[2.0, 2.0], [2.0, 2.0]]
