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
            import utils.factory
            import evaluation.eval_wrapper
            import evaluation.tusimple.lane
        return SimpleNamespace(
            parsingNet=model.model.parsingNet,
            constant=data.constant,
            dataset=data.dataset,
            mytransforms=data.mytransforms,
            loss=utils.loss,
            metrics=utils.metrics,
            factory=utils.factory,
            eval_wrapper=evaluation.eval_wrapper,
            lane=evaluation.tusimple.lane,
            root=REF,
        )
    finally:
        sys.path.remove(str(REF))
