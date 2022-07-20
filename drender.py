import torch
from d_render import depth_pcloud_render, depth_pcloud_render_idx
from tqdm import tqdm

class DepthRender:
    def __init__(self):
        pass

    def forward(self, pcloud, zbuf, idbuf=None):
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
                c_idbuf = torch.zeros_like(idbuf).squeeze(1).int().contiguous()
                depth_pcloud_render_idx(c_pcloud, c_zbuf, c_idbuf)
            zbuf[:, 0] = c_zbuf.float() / 1e5
            idbuf[:, 0] = c_idbuf
            return zbuf, idbuf

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
