import torch
from remadom.utils.schedules import WarmupCosine

def test_warmup_cosine_schedule_increases_then_decays():
    lin = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    sched = WarmupCosine(opt, base_lr=0.1, warmup_steps=5, total_steps=20, min_lr=0.01)
    lrs = []
    for _ in range(20):
        sched.step()
        lrs.append(sched.get_lr())
    assert lrs[0] < lrs[4]  # warmup increases
    assert lrs[-1] <= 0.11 and lrs[-1] >= 0.009  # near min_lr