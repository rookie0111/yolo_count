import torch
torch.cuda.empty_cache()
print('Testing CUDA...', flush=True)
t = torch.randn(1000, 1000).cuda()
print(f'Basic tensor OK: {t.shape}', flush=True)
for _ in range(20):
    a = torch.randn(1024, 1024).cuda()
    b = torch.randn(1024, 1024).cuda()
    c = a @ b
    del a, b, c
    torch.cuda.empty_cache()
print('GEMM test OK', flush=True)

for _ in range(10):
    x = torch.randn(4, 256, 64, 64).cuda().requires_grad_()
    w = torch.randn(256, 256).cuda().requires_grad_()
    y = (x * w.view(1, -1, 1, 1)).sum()
    y.backward()
    del x, w, y
    torch.cuda.empty_cache()
print('Backward test OK', flush=True)

for r in [16, 32, 64, 128]:
    for _ in range(5):
        x = torch.randn(1, 16384, 256).cuda()
        lora_a = torch.randn(1, r, 256).cuda()
        lora_b = torch.randn(1, 256, r).cuda()
        out = x @ lora_a.transpose(1, 2)
        out = out @ lora_b.transpose(1, 2)
        del x, lora_a, lora_b, out
        torch.cuda.empty_cache()
    print(f'LoRA rank={r} GEMM OK', flush=True)
print('ALL CUDA TESTS PASSED', flush=True)
