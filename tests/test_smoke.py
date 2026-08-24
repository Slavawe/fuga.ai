import sys
import torch

sys.path.insert(0, ".")

from antitf.vsa import VSAEncoder
from antitf.adapter import VectorAdapter, SequencePooler
from antitf.kan import ChebyKANLayer, ChebyMLP
from antitf.jepa import HJEPA, HJEPAPredictor
from antitf.vicreg import VICRegLoss
from antitf.owm import OWMExecutor, WoodburyOWMExecutor
from antitf.ast_vsa import TreeSitterVSAEncoder
from antitf.data import make_windows, CORPUS_TASK_0


def test_vsa():
    vsa = VSAEncoder(vocab_size=256, hyper_dim=1024)
    windows, raws = make_windows(CORPUS_TASK_0[:2])
    hv = vsa(windows)
    assert hv.shape == (windows.shape[0], 1024)
    assert set(hv.unique().tolist()) <= {-1.0, 1.0}
    syn = vsa.encode_syntax(raws)
    mixed = vsa.mix(hv, syn)
    assert mixed.shape == hv.shape


def test_kan():
    layer = ChebyKANLayer(8, 4, degree=4)
    x = torch.randn(5, 8)
    assert layer(x).shape == (5, 4)


def test_jepa_and_loss():
    model = HJEPA(hyper_dim=256, latent_dim=32)
    seq = torch.randn(6, 10, 256).sign()
    out = model(seq)
    loss = VICRegLoss()(out["pred_l0"], out["target_l0"])
    loss.backward()
    assert torch.isfinite(loss)


def test_owm_projection():
    for cls in (OWMExecutor, WoodburyOWMExecutor):
        lin = torch.nn.Linear(4, 3, bias=False)
        holder = torch.nn.ModuleDict({"lin": lin})
        owm = cls(holder, lr=0.1)
        A_old = torch.randn(64, 4)
        x_old = torch.randn(16, 4)
        y_before = lin(x_old).detach()
        owm.update_space("lin.weight", A_old)
        for _ in range(20):
            y = lin(x_old)
            ((y - y_before) ** 2).sum().backward()
            owm.apply_gradients(lr=0.1)
            owm.zero_grad()
        drift = (lin(x_old) - y_before).abs().max().item()
        base = y_before.abs().max().item() + 1e-9
        assert drift / base < 0.05, f"{cls.__name__} failed to protect old task: {drift}"


def test_woodbury_matches_direct():
    eps = 1e-3
    A = torch.randn(8, 64)
    n = A.shape[0]
    inner = (A @ A.T) / eps + torch.eye(n)
    proj_factorized = (A.T @ torch.linalg.inv(inner) @ A) / eps
    ridge = A.T @ torch.linalg.inv(A @ A.T + eps * torch.eye(n)) @ A
    assert torch.allclose(proj_factorized, ridge, atol=1e-3)


def test_ast_encoder():
    enc = TreeSitterVSAEncoder(hyper_dim=512)
    code_a = b"int add(int a, int b) { return a + b; }"
    code_b = b"int add(int a, int b) { return a - b; }"
    code_c = b"struct Node { int x; };"
    ha, hb, hc = enc.encode_batch([code_a, code_b, code_c])
    assert ha.shape == (512,)
    assert set(ha.unique().tolist()) <= {-1.0, 1.0}
    def sim(u, v):
        return (u * v).float().mean().item()
    assert sim(ha, hb) > sim(ha, hc), "similar ASTs should be more alike"


if __name__ == "__main__":
    test_vsa()
    test_kan()
    test_jepa_and_loss()
    test_owm_projection()
    test_woodbury_matches_direct()
    test_ast_encoder()
    print("all tests passed")
