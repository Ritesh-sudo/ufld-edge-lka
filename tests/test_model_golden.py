"""Golden test: the ported model must equal the original, weight for weight."""
import pytest
import torch

from ufld.model import ParsingNet

CLS_DIM = (101, 56, 4)


def _pair(ref, backbone, use_aux):
    torch.manual_seed(0)
    ref_net = ref.parsingNet(pretrained=False, backbone=backbone, cls_dim=CLS_DIM, use_aux=use_aux)
    our_net = ParsingNet(backbone=backbone, pretrained=False, cls_dim=CLS_DIM, use_aux=use_aux)
    # strict load: keys must match one-for-one (this is what lets the authors' checkpoints load)
    our_net.load_state_dict(ref_net.state_dict(), strict=True)
    return ref_net, our_net


@pytest.mark.parametrize('backbone', ['18', '34'])
@pytest.mark.parametrize('use_aux', [False, True])
@pytest.mark.parametrize('train_mode', [False, True])
def test_outputs_match_reference(ref, backbone, use_aux, train_mode):
    ref_net, our_net = _pair(ref, backbone, use_aux)
    ref_net.train(train_mode)
    our_net.train(train_mode)
    x = torch.randn(2, 3, 288, 800, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want, got = ref_net(x), our_net(x)
    if use_aux:
        torch.testing.assert_close(got[0], want[0], rtol=0, atol=1e-5)
        torch.testing.assert_close(got[1], want[1], rtol=0, atol=1e-5)
        assert got[0].shape == (2, *CLS_DIM) and got[1].shape == (2, 5, 36, 100)
    else:
        torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
        assert got.shape == (2, *CLS_DIM)


def test_outputs_are_not_trivially_equal(ref):
    # guards against a vacuous pass (e.g. both outputs all zeros)
    _, our_net = _pair(ref, '18', False)
    our_net.eval()
    out = our_net(torch.randn(1, 3, 288, 800))
    assert out.std() > 1e-4


def test_inference_model_has_no_aux_parameters():
    net = ParsingNet(use_aux=False)
    assert not any(k.startswith('aux_') for k in net.state_dict())


def test_head_dim_follows_input_size():
    net = ParsingNet(size=(320, 640), use_aux=False)
    assert net.fea_dim == 8 * 10 * 20
    assert net(torch.randn(1, 3, 320, 640)).shape == (1, *CLS_DIM)
