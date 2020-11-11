import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: remove this dependency
from torchsearchsorted import searchsorted


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.maxes = torch.tensor([-9999.0 for _ in range(90)])
        self.mins = torch.tensor([9999.0 for _ in range(90)])
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, track_values=True):
        if track_values:
            with torch.no_grad():
                max = x.max(dim=0)[0]
                min = x.min(dim=0)[0]
                self.maxes = torch.stack([self.maxes, max]).max(dim=0)[0]
                self.mins = torch.stack([self.mins, min]).min(dim=0)[0]
    
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        self.hidden_states = []
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            self.hidden_states.append(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                self.hidden_states.append(h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))
        
class HyperNeRF(nn.Module):
    def __init__(self, nerf, Class_dim = 2, Z_dim = 16, C_dim = 1, verbose=False, dev='cpu'):
        super(HyperNeRF, self).__init__()
        self.D = nerf.D
        self.W = nerf.W
        self.input_ch = nerf.input_ch
        self.input_ch_views = nerf.input_ch_views
        self.skips = nerf.skips
        self.use_viewdirs = nerf.use_viewdirs

        # Replace with actual latent variable
        #self.Z = torch.rand(Z_dim)
        self.Class_dim = Class_dim
        self.Class = torch.zeros(Class_dim).to(dev)

        weights_and_biases = []
        for name,param in nerf.named_parameters():
            if "bias" in name:
                weights_and_biases[-1] = torch.cat((weights_and_biases[-1], param[...,None]), dim=1)
            else:
                weights_and_biases.append(param)

        self.target_shape = []
        weight_gen_sizes = []
        final_size = 0
        for (i,w_and_b) in enumerate(weights_and_biases):
            self.target_shape.append(w_and_b.shape)
            weight_gen_sizes.append(torch.Size([w_and_b.shape[0], C_dim]))

        if verbose:
            print("Target size\n----------------------")
        for shape in self.target_shape:
            if verbose:
                print(shape)

        if verbose:
            print("----------------------")
        if verbose:
            print("C size\n----------------------")

        C = torch.Tensor(weight_gen_sizes[0])
        if verbose:
            print(weight_gen_sizes[0])
        for shape in weight_gen_sizes[1:]:
            if verbose:
                print(shape)
            C = torch.cat((C, torch.Tensor(shape)), dim=0)

        self.C_shape = C.shape

        if verbose: print(" -- OR -- \n", self.C_shape, "\n")

        self.net_E = nn.Sequential(
                nn.Linear(Class_dim, Class_dim*2),
                nn.ReLU(inplace=True),
                nn.Linear(Class_dim*2, Class_dim*4),
                nn.ReLU(inplace=True),
                nn.Linear(Class_dim*4, self.C_shape[0] * C_dim)
        )

        self.linears = nn.ModuleList()
        for size in self.target_shape:
            t_len = size[1]
            self.linears.append(nn.Linear(size[0] * C_dim, size[0]*t_len))
            
    def NeRF_Functional(self, X, Weights, verbose=False):
        input_pts, input_views = torch.split(X, [self.input_ch, self.input_ch_views], dim=-1)
        H = input_pts
        for i in range(0, self.D):
            w = Weights[i][:,:-1]
            b = Weights[i][:,-1]
            H = F.relu(F.linear(H, w, bias=b))
            if i in self.skips:
                H = torch.cat([input_pts, H], -1)

        VIEW_LINEAR = [8]
        ALPHA = 10
        FEATURE = 9
        RGB = 11

        if self.use_viewdirs:
            alpha = F.linear(H, Weights[ALPHA][:,:-1], bias=Weights[ALPHA][:,-1])
            feature = F.linear(H, Weights[FEATURE][:,:-1], bias=Weights[FEATURE][:,-1])
            H = torch.cat([feature, input_views], -1)

            for i in VIEW_LINEAR:
                H = F.relu(F.linear(H, Weights[i][:,:-1], bias=Weights[i][:,-1]))

            rgb = F.linear(H, Weights[RGB][:,:-1], bias=Weights[RGB][:,-1])
            outputs = torch.cat([rgb, alpha], -1)
        else:
            OUTPUT_LINEAR = 9
            outputs = F.linear(H, Weights[OUTPUT_LINEAR][:,:-1], bias=Weights[OUTPUT_LINEAR][:,-1])

        return outputs

    def forward(self, X, verbose=False):
        total_params = 0
        if verbose:
            print("Input", self.Z.shape, sep='\t\t|\t\t')

        C = self.Class#Z
        for layer in self.net_E:
            C = layer(C)
            num_params = sum([p.numel() for p in layer.parameters()])
            total_params += num_params
            if verbose:
                print(layer.__class__.__name__, C.shape, num_params, sep='\t\t|\t\t')

        weights = []
        c_idx = 0
        for size, linear in zip(self.target_shape, self.linears):
            t_len = size[0]
            C_sub = C[c_idx:c_idx+t_len*self.C_shape[1]]
            weights.append(linear(C_sub).view(size))
            c_idx += t_len

            num_params = sum([p.numel() for p in linear.parameters()])
            total_params += num_params
            if verbose:
                print(linear.__class__.__name__, weights[-1].shape, num_params, sep='\t\t|\t\t')

        if verbose:
            print("Total Parameters:", total_params, '\n')

        return self.NeRF_Functional(X, weights, verbose=verbose)




# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
