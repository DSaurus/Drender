import torch
from torch.autograd import Function
from d_render import depth_pcloud_render, depth_pcloud_render_idx, depth_pcloud_render_idx_backward
from tqdm import tqdm

class FastDepthRender(Function):
    def __init__(self):
        super(Function, self).__init__()
    
    @staticmethod
    def forward(ctx, pcloud, zbuf, idbuf=None):
        # pcloud B C N
        # zbuf B C H W
        if idbuf is None:
            with torch.no_grad():
                c_pcloud = pcloud.permute(0, 2, 1).contiguous()
                c_zbuf = torch.zeros_like(zbuf).squeeze(1).int().contiguous()
                depth_pcloud_render(c_pcloud, c_zbuf)
            zbuf[:, 0] = c_zbuf.float() / 1e5
            return zbuf
        else:
            with torch.no_grad():
                c_pcloud = pcloud.permute(0, 2, 1).contiguous()
                c_zbuf = torch.zeros_like(zbuf).squeeze(1).int().contiguous()
                c_idbuf = torch.zeros_like(idbuf).squeeze(1).int().contiguous() - 1
                depth_pcloud_render_idx(c_pcloud, c_zbuf, c_idbuf)
            zbuf[:, 0] = c_zbuf.float() / 1e5
            idbuf[:, 0] = c_idbuf
            return zbuf, idbuf

class DepthRender(Function):

    def __init__(self):
        super(Function, self).__init__()

    @staticmethod
    def forward(ctx, pcloud, zbuf, idbuf):
        # pcloud B C N
        # zbuf B C H W
        c_pcloud = pcloud.permute(0, 2, 1).contiguous()
        c_zbuf = torch.zeros_like(zbuf).squeeze(1).int().contiguous()
        c_idbuf = torch.zeros_like(idbuf).squeeze(1).int().contiguous() - 1
        depth_pcloud_render_idx(c_pcloud, c_zbuf, c_idbuf)
        zbuf[:, 0] = c_zbuf.float() / 1e5
        idbuf[:, 0] = c_idbuf

        ctx.save_for_backward(c_pcloud, c_idbuf)
        return zbuf

    @staticmethod
    def backward(ctx, grad_input):
        
        pcloud, idbuf = ctx.saved_tensors
        B, N, C = pcloud.shape
        B, H, W = idbuf.shape
        c_grad_input = grad_input.squeeze(1).contiguous()
        grad_output = torch.zeros_like(pcloud)
        c_grad_output = grad_output[:, :, 2].contigous()
        depth_pcloud_render_idx_backward(pcloud, idbuf, c_grad_input, c_grad_output)

        grad_output[:, :, 2] = c_grad_output
        return grad_output.permute(0, 2, 1), None, None
        
        return grad_output1, grad_output2.permute(0, 3, 1, 2), None, None, None

class DepthRender:
    def __init__(self):
        pass

    

if __name__ == "__main__":
    depth_render = DepthRender()
    pcloud = torch.randn((2, 3, 256*256)).cuda() / 2
    zbuf = torch.zeros((2, 1, 256, 256)).cuda()
    for i in tqdm(range(1000)):
        zbuf = depth_render.forward(pcloud, zbuf)
    zbuf = (zbuf + 2).clamp(0, 4)
    zbuf = zbuf[0, 0].detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.imshow(zbuf)
    plt.savefig('test.jpg')
