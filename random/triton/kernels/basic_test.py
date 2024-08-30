import torch
import triton
import triton.language as tl

print(f"torch version: {torch.__version__}")
print(f"triton version: {triton.__version__}")

@triton.jit
def simple_kernel(in_ptr0, out_ptr0, xnumel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(1)

    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, mask=xmask)
    tl.store(out_ptr0+xindex, tmp0, mask=xmask)


def test_simple_kernel():
    DATA_SIZE = 1024
    BLOCK_SIZE = 32

    inp = torch.randn(DATA_SIZE, device='cuda')
    outp = torch.zeros(DATA_SIZE, device='cuda')

    # compute grid size
    grid = (DATA_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    # launch kernel
    simple_kernel[(grid,)](inp, outp, DATA_SIZE, BLOCK_SIZE=BLOCK_SIZE)

    torch.testing.assert_close(inp, outp)
    print(f"Success! Output matches Input.")


if __name__ == '__main__':
    test_simple_kernel()