"""Golden tests compare the port against the original repo cloned in reference/.

The original uses top-level package names (model, data, utils, evaluation), so it is
imported by putting reference/Ultra-Fast-Lane-Detection first on sys.path. The tests are
skipped when the clone is absent (reference/ is gitignored).
"""
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

REF = Path(__file__).resolve().parents[1] / 'reference' / 'Ultra-Fast-Lane-Detection'


@pytest.fixture(scope='session')
def ref():
    if not REF.exists():
        pytest.skip('reference/Ultra-Fast-Lane-Detection not cloned')
    sys.path.insert(0, str(REF))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # reference uses deprecated torchvision `pretrained=`
            import data.constant
            import data.dataset
            import data.mytransforms
            import model.model
            import utils.loss
            import utils.metrics
        return SimpleNamespace(
            parsingNet=model.model.parsingNet,
            constant=data.constant,
            dataset=data.dataset,
            mytransforms=data.mytransforms,
            loss=utils.loss,
            metrics=utils.metrics,
            root=REF,
        )
    finally:
        sys.path.remove(str(REF))
