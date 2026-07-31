import pytest

from jgrec.core.cuda import require_jittor_cuda


class _Flags:
    use_cuda = 0


class _Jittor:
    def __init__(self, has_cuda: bool) -> None:
        self.has_cuda = has_cuda
        self.flags = _Flags()


def test_require_jittor_cuda_enables_cuda_and_rejects_cpu_only_runtime():
    available = _Jittor(has_cuda=True)

    require_jittor_cuda(available)

    assert available.flags.use_cuda == 1

    unavailable = _Jittor(has_cuda=False)
    with pytest.raises(RuntimeError, match="CUDA"):
        require_jittor_cuda(unavailable)
