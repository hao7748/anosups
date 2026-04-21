import sys
from pathlib import Path
import torch
import numpy as np
import random

MAE_DIR = Path(__file__).resolve().parent.parent / "mae"
if str(MAE_DIR) not in sys.path:
    sys.path.insert(0, str(MAE_DIR))

import models_mae


def upsample_patches(big_patch_values):
    """Upsample coarse patch scores (length 1/4/49) to 196 fine-grained patches."""
    in_len = len(big_patch_values)
    small_grid = 14
    out = np.zeros(196)

    if in_len == 196:
        return big_patch_values.copy()
    elif in_len == 49:  # 7x7 grid, each coarse patch covers 2x2 fine patches
        big_grid = 7
        ratio = 2
    elif in_len == 4:   # 2x2 grid, each coarse patch covers 7x7 fine patches
        big_grid = 2
        ratio = 7
    elif in_len == 1:   # whole image as single patch
        out[:] = big_patch_values[0]
        return out
    else:
        raise ValueError("in_len must be 1, 4, 49, or 196")

    for i in range(big_grid):
        for j in range(big_grid):
            value = big_patch_values[i * big_grid + j]
            for r in range(ratio):
                for c in range(ratio):
                    idx = (i * ratio + r) * small_grid + (j * ratio + c)
                    out[idx] = value
    return out


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def per_patch_loss(model, imgs, pred, mask):
    target = model.patchify(imgs)
    if model.norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)
    return loss


def random_masking(x, mask_ratio, index):
    N, L, D = x.shape

    noise = torch.rand(N, L, device=x.device)
    index_tensor = torch.as_tensor(index, device=x.device, dtype=noise.dtype)
    if index_tensor.numel() != L:
        raise ValueError(f"index length {index_tensor.numel()} does not match L={L}")
    noise[0].copy_(index_tensor)
    n0 = np.count_nonzero(index)
    len_keep = int(L * (1 - mask_ratio)) - n0

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def run_one_image(img, model, index, mask_ratio):
    x = torch.tensor(img).unsqueeze(0)
    x = torch.einsum('nhwc->nchw', x)

    latent = model.patch_embed(x.float()) + model.pos_embed[:, 1:, :]
    latent, mask, ids_restore = random_masking(latent, mask_ratio, index)

    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(latent.shape[0], -1, -1)
    latent = torch.cat((cls_tokens, latent), dim=1)

    for blk in model.blocks:
        latent = blk(latent)
    latent = model.norm(latent)

    pred = model.forward_decoder(latent, ids_restore)
    y = model.unpatchify(pred)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)
    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask

    return (
        x[0].permute(2, 0, 1).view(1, 3, 224, 224).cpu().data,
        im_paste[0].permute(2, 0, 1).view(1, 3, 224, 224).cpu().data,
        im_masked[0].permute(2, 0, 1).view(1, 3, 224, 224).cpu().data,
    )


def _encode_and_loss(img, model, index, mask_ratio):
    """Single MAE forward pass; returns per-patch loss and patch-level mask."""
    x = torch.tensor(img).unsqueeze(0)
    x = torch.einsum('nhwc->nchw', x)

    latent = model.patch_embed(x.float()) + model.pos_embed[:, 1:, :]
    latent, mask, ids_restore = random_masking(latent, mask_ratio, index)

    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    latent = torch.cat((cls_token.expand(latent.shape[0], -1, -1), latent), dim=1)

    for blk in model.blocks:
        latent = blk(latent)
    latent = model.norm(latent)

    pred = model.forward_decoder(latent, ids_restore)
    patch_loss = per_patch_loss(model, x.float(), pred, mask)
    return patch_loss, mask


def getsim(patchsnumer, img, model, mask_ratio, ss):
    index = upsample_patches(np.random.random_sample(patchsnumer))

    patch_loss1, mask1 = _encode_and_loss(img, model, index, mask_ratio)
    patch_loss2, mask2 = _encode_and_loss(img, model, 1 - index, mask_ratio)

    ssimindex = patch_loss2[0] * mask2[0] + patch_loss1[0] * abs(1 - mask2[0])
    ssimindex = np.where(ssimindex.detach().numpy() > ss, 1, 0)
    return np.array(ssimindex)


def getindex(num_imgs):
    n = num_imgs
    indexarray = np.zeros((n, 196))
    oneindex = np.arange(196)
    random.shuffle(oneindex)

    for i in range(n):
        p = i * int(196 / n)
        q = 196 if i == n - 1 else (i + 1) * int(196 / n)
        for j in range(p, q):
            indexarray[i][oneindex[j]] = 1

    return indexarray  # (n, 196)
