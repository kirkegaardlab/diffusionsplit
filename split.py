import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from torch import vmap
from torch.nn.functional import binary_cross_entropy_with_logits
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import sys
from tensorboardX import SummaryWriter
import pickle
import time

from data.dataset import DiffSplitDataset
import cpunet
nbase = (32, 64, 128, 256)  # The UNet architecture.

try:
    from cc_torch import connected_components_labeling
    do_component_split = True
except ImportError:
    connected_components_labeling = None
    do_component_split = False
    print('Could not find cc_torch. Code running without connected component splitting'
          ' (slow and VRAM intensive inference)')
    print('Install from https://github.com/zsef123/Connected_components_PyTorch')

def load_dataset(dataset, train):
    dataset = DiffSplitDataset(direc=dataset, test=not train, augment=train, max_n=None)
    return dataset


def train_segmentation(model_name, dataset, epochs, batch_size, load, plot, info=None):
    # Settings
    device = dataset[0][0].device
    Path(f'models/{model_name}').mkdir(parents=True, exist_ok=True)

    # Make data loader
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )

    writer = SummaryWriter(f'models/{model_name}/log_segmentation')

    # Make model
    in_channels = dataset[0][0].shape[0]
    out_channels = 1  # segmentation
    model = cpunet.create_net(in_channels, out_channels, nbase=nbase)
    if load and Path(f'models/{model_name}/segmentation.pt').exists():
        model.load_state_dict(torch.load(f'models/{model_name}/segmentation.pt', map_location='cpu'))
    model.to(device)
    model.train()

    # Make optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    if load and Path(f'models/{model_name}/segmentation_opt.pt').exists():
        opt.load_state_dict(torch.load(f'models/{model_name}/segmentation_opt.pt', map_location='cpu'))

    # Train
    if plot:
        last_plot = 0
        plt.figure(figsize=(12, 12))
    loss_term = torch.nn.BCEWithLogitsLoss()
    for epoch in tqdm(range(epochs), desc='Epoch', position=0):
        avg_loss = []
        for im, labels in tqdm(loader, leave=False, position=1):
            label_segmentation = labels.max(dim=3)[0][:, None, ...]
            predicted = model(im)
            loss = loss_term(predicted, 1.0 * label_segmentation)
            avg_loss.append(float(loss))
            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f'\nEpoch {epoch} : loss = {np.mean(avg_loss):.4f} with std {np.std(avg_loss):.4f}')
        writer.add_scalar('avg_loss', np.mean(avg_loss), epoch)

        if plot and (time.time() - last_plot) > 5:
            last_plot = time.time()
            with torch.no_grad():
                plt.clf()
                for i in range(4):
                    plt.subplot(2, 4, i + 1)
                    plt.imshow(label_segmentation[i, 0].cpu())
                    plt.axis('off')
                    plt.subplot(2, 4, i + 5)
                    plt.imshow((predicted[i, 0] > 0.0).cpu())
                    plt.axis('off')
                plt.pause(0.001)

    # Save
    torch.save(model.state_dict(), f'models/{model_name}/segmentation.pt')
    torch.save(opt.state_dict(), f'models/{model_name}/segmentation_opt.pt')
    if info:
        with open(f'models/{model_name}/segmentation_args.txt', 'w') as f:
            json.dump(info, f, indent=2)


def calc_optimum(spl1, spl2, labels):
    """
    Given two noisy images (spl1, spl2) and labels
    this function returns two images with the labels split
    into two such that a label ends up in the split with
    the highest noise sum.
    """
    s_1 = -torch.sum(labels * spl1[:, :, :, None], dim=(1, 2))
    s_2 = -torch.sum(labels * spl2[:, :, :, None], dim=(1, 2))
    mask = 1 * (s_1 < s_2)

    s_1 = torch.sum(labels * mask[:, None, None, :], dim=3) > 0.5
    s_2 = torch.sum(labels * (1 - mask)[:, None, None, :], dim=3) > 0.5

    return s_1, s_2


def train_split(model_name, dataset, epochs, batch_size, load, plot, info=None):
    # Settings
    device = dataset[0][0].device
    Path(f'models/{model_name}').mkdir(parents=True, exist_ok=True)
    Path(f'models/{model_name}/log').mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(f'models/{model_name}/log')

    # Make data loader
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )

    # Make model
    in_channels = dataset[0][0].shape[0] + 1 + 2  # segmentation + a, b
    out_channels = 2  # a, b
    model = cpunet.create_net(in_channels, out_channels, nbase=nbase)
    if load and Path(f'models/{model_name}/split.pt').exists():
        model.load_state_dict(torch.load(f'models/{model_name}/split.pt', map_location='cpu'))
    model.to(device)
    model.train()

    # Make optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if load and Path(f'models/{model_name}/split_opt.pt').exists():
        opt.load_state_dict(torch.load(f'models/{model_name}/split_opt.pt', map_location='cpu'))

    # Load segmentation model
    in_channels = dataset[0][0].shape[0]
    seg_model = cpunet.create_net(in_channels, 1, nbase=nbase)
    seg_model.load_state_dict(torch.load(f'models/{model_name}/segmentation.pt', map_location='cpu'))
    seg_model.to(device)
    seg_model.eval()

    # Train
    for epoch in tqdm(range(epochs), desc='Epoch', position=0):
        avg_loss = []
        im_i = -1
        for im, labels in loader:
            im_i += 1
            orig_label_segmentation = labels.max(dim=3)[0]

            # Remove labels
            for i_batch in range(labels.shape[0]):
                if np.random.random() < 0.5:
                    p = np.random.random()  # sampled remove probability
                    labels[i_batch, :, :, np.random.random(labels.shape[-1]) < p] = 0

            label_segmentation = labels.max(dim=3)[0]

            # Run segmentation model:
            with torch.no_grad():
                segmentation = seg_model(im) > 0.0
                segmentation[label_segmentation[:, None, ...] == 0] = 0

            # Initialize a, b
            a = (torch.randn(segmentation.shape, device=device) * segmentation)[:, 0, ...]
            b = (torch.randn(segmentation.shape, device=device) * segmentation)[:, 0, ...]
            opt_a, opt_b = calc_optimum(a, b, labels)

            # Choose times
            t = 10**(-10 * (1 - torch.rand(a.shape[0], device=device)**2) * torch.rand(a.shape[0], device=device))
            t[torch.rand(segmentation.shape[0], device=device) < 0.35] = 0.0  # sometimes go completely to zero

            # Update a, b
            do = 'diffusion' if np.random.random() < 0.5 else 'bootstrap'
            if do == 'diffusion':
                a = t[:, None, None] * opt_a + (1 - t)[:, None, None] * a
                b = t[:, None, None] * opt_b + (1 - t)[:, None, None] * b
            elif do == 'bootstrap':
                # Sometimes we do bootstrapping on the noise instead of simply sampling a `t`.
                # This can help the early steps where the input is not necessarily a linear combination
                # the labels and gaussian noise.
                with torch.no_grad():
                    full_inp = torch.cat((im, segmentation, a[:, None, :, :], b[:, None, :, :]), dim=1)
                    res = torch.sigmoid(model(full_inp))

                    bootstrap_a = res[:, 0, :, :] * segmentation[:, 0, ...]
                    bootstrap_b = res[:, 1, :, :] * segmentation[:, 0, ...]

                    a = t[:, None, None] * bootstrap_a + (1 - t)[:, None, None] * a
                    b = t[:, None, None] * bootstrap_b + (1 - t)[:, None, None] * b

                    # Optimum could have changed, so update:
                    opt_a, opt_b = calc_optimum(a, b, labels)
            else:
                raise AssertionError()

            # Predict
            full_inp = torch.cat((im, segmentation, a[:, None, :, :], b[:, None, :, :]), dim=1)
            predicted = model(full_inp)

            # Loss
            loss_a = binary_cross_entropy_with_logits(predicted[:, 0, ...], 1.0 * opt_a, reduction='none')
            loss_b = binary_cross_entropy_with_logits(predicted[:, 1, ...], 1.0 * opt_b, reduction='none')
            loss = torch.sum(segmentation * (loss_a + loss_b)) / (0.1 + torch.sum(segmentation))
            avg_loss.append(float(loss))

            # Step
            loss.backward()
            opt.step()
            opt.zero_grad()

            if plot and epoch >= 1 and im_i % 10 == 0:
                plt.pause(0.001)

        print(f'\nEpoch {epoch} : loss = {np.mean(avg_loss):.4f} with std {np.std(avg_loss):.4f}')
        writer.add_scalar('avg_loss', np.mean(avg_loss), epoch)

        if plot:
            with torch.no_grad():
                plt.clf()
                plt.subplot(2, 4, 1)
                plt.title(f'{do}')
                plt.axis('off')
                imshow = np.transpose(im[0].cpu().numpy(), [1, 2, 0])
                if imshow.shape[2] == 2:
                    imshow = np.concatenate((imshow, np.zeros((imshow.shape[0], imshow.shape[1], 1))), axis=2)
                plt.imshow(imshow)
                plt.subplot(2, 4, 5)
                plt.axis('off')
                plt.title(f'{t[0] : .3f}')
                plt.imshow((orig_label_segmentation[0] + label_segmentation[0]).cpu(), vmin=0, vmax=2)
                plt.subplot(2, 4, 2)
                plt.axis('off')
                plt.imshow(a[0].cpu(), vmin=-2, vmax=2)
                plt.subplot(2, 4, 6)
                plt.axis('off')
                plt.imshow(b[0].cpu(), vmin=-2, vmax=2)
                plt.subplot(2, 4, 3)
                plt.axis('off')
                plt.imshow(opt_a[0].cpu(), vmin=0, vmax=1)
                plt.subplot(2, 4, 7)
                plt.axis('off')
                plt.imshow(opt_b[0].cpu(), vmin=0, vmax=1)
                plt.subplot(2, 4, 4)
                plt.axis('off')
                plt.imshow((segmentation[0, 0] * predicted[0, 0]).cpu() > 0, vmin=0, vmax=1)
                plt.subplot(2, 4, 8)
                plt.axis('off')
                plt.imshow((segmentation[0, 0] * predicted[0, 1]).cpu() > 0, vmin=0, vmax=1)

                if epoch == 0:
                    plt.pause(0.001)
                else:
                    plt.pause(0.001)

                writer.add_figure('example', plt.gcf(), epoch)

        # Update learning rate
        for g in opt.param_groups:
            g['lr'] = 0.05**(1/epochs) * g['lr']  # epochs over which we go down two orders of magnitude

    # Save
    torch.save(model.state_dict(), f'models/{model_name}/split.pt')
    torch.save(opt.state_dict(), f'models/{model_name}/split_opt.pt')
    if info:
        with open(f'models/{model_name}/split_args.txt', 'w') as f:
            json.dump(info, f, indent=2)

    writer.close()


def split(device, inp, net, seg, time_steps):
    inp = torch.cat((inp, seg), dim=1)

    a = (torch.randn((inp.shape[0], 1, inp.shape[2], inp.shape[3]), device=device)
                  * inp[:, -1, ...][:, None, ...])
    b = (torch.randn((inp.shape[0], 1, inp.shape[2], inp.shape[3]), device=device)
                  * inp[:, -1, ...][:, None, ...])

    for t_i in range(1, len(time_steps)):
        with torch.no_grad():
            full_inp = torch.cat((inp, a, b), dim=1)
            t = time_steps[t_i]

            pred = net(full_inp)
            predicted = torch.sigmoid(pred)

            tp = time_steps[t_i - 1]
            a = predicted[:, 0, ...][:, None, ...] * (t - tp) / (1 - tp) + a * (1 - t) / (1 - tp)
            b = predicted[:, 1, ...][:, None, ...] * (t - tp) / (1 - tp) + b * (1 - t) / (1 - tp)

            a = a * seg[:, :1]
            b = b * seg[:, :1]

    return a, b


def segment(c):
    a = connected_components_labeling(c)

    vs, idxs = torch.unique(a, return_inverse=True, sorted=True)
    k = torch.arange(len(vs), device=vs.device)
    a = k[idxs]

    return a


def comp_idxs(a):
    ar = torch.sum(a, dim=0) > 0
    r1, r2 = torch.argmax(1 * ar), len(ar) - torch.argmax(torch.flip(1 * ar, dims=(0, )))
    ac = torch.sum(a, dim=1) > 0
    c1, c2 = torch.argmax(1 * ac), len(ac) - torch.argmax(torch.flip(1 * ac, dims=(0,)))
    idxs = torch.stack((c1, c2, r1, r2))
    return idxs

def calc_dimensions(comp):
    m = torch.arange(1, comp.max() + 1, device=comp.device)
    a = comp[None, ...] == m[:, None, None]

    idxs = vmap(comp_idxs)(a)

    rsize = idxs[:, 1] - idxs[:, 0]
    csize = idxs[:, 3] - idxs[:, 2]

    return idxs, len(m), torch.max(rsize), torch.max(csize)



def component_split(inp, seg, comp_split_idxs, zero_threshold):
    """
    Standard connected components segmentation. We do this to reduce the image resolution
    to the smallest possible size.
    """
    c_seg = seg.to(torch.uint8)
    comps = []
    idxs = []
    lengths = torch.empty(c_seg.shape[0], device=seg.device, dtype=torch.int64)
    sizes = torch.empty((c_seg.shape[0], 2), device=seg.device, dtype=torch.int64)
    for i in range(c_seg.shape[0]):
        comp = segment(c_seg[i, 0])
        comps.append(comp)
        idx, length, rsize, csize = calc_dimensions(comp)
        idxs.append(idx)
        lengths[i] = length
        sizes[i, 0] = rsize
        sizes[i, 1] = csize
    rmax = torch.max(sizes[:, 0])
    cmax = torch.max(sizes[:, 1])
    r_dim = rmax + (16 - rmax % 16)
    c_dim = cmax + (16 - cmax % 16)
    achievable_dim = torch.sum(c_dim * r_dim * lengths)
    current_dim = seg.shape[0] * seg.shape[2] * seg.shape[3]
    potential_speedup = current_dim / achievable_dim

    if potential_speedup > 1.1:
        print(f'Component splitting!')

        new_inp = []
        new_seg = []
        new_comp_split_idxs = []
        for i, idx in enumerate(idxs):
            r_dist = idx[:, 1] - idx[:, 0]
            add = (r_dim - r_dist) // 2
            idx[:, 0] = torch.maximum(0 * idx[:, 0], idx[:, 0] - add)
            idx[:, 1] = idx[:, 0] + r_dim
            sub = torch.maximum(0 * idx[:, 1], idx[:, 1] - c_seg.shape[2])
            idx[:, :2] -= sub[:, None]

            c_dist = idx[:, 3] - idx[:, 2]
            add = (c_dim - c_dist) // 2
            idx[:, 2] = torch.maximum(0 * idx[:, 2], idx[:, 2] - add)
            idx[:, 3] = idx[:, 2] + c_dim
            sub = torch.maximum(0 * idx[:, 2], idx[:, 3] - c_seg.shape[3])
            idx[:, 2:] -= sub[:, None]

            for j in range(len(idx)):
                new_inp.append(inp[i, :, idx[j, 0]:idx[j, 1], idx[j, 2]:idx[j, 3]][None])
                new_seg.append(comps[i][idx[j, 0]:idx[j, 1], idx[j, 2]:idx[j, 3]][None, None] == (j + 1))
                new_comp_split_idxs.append((comp_split_idxs[i] + idx[j, ::2])[None])

        inp = torch.concatenate(new_inp, dim=0)
        seg = torch.concatenate(new_seg, dim=0)
        comp_split_idxs = torch.concatenate(new_comp_split_idxs, dim=0)

        s = torch.sum(seg, dim=(1, 2, 3))
        seg = seg[s > zero_threshold]
        inp = inp[s > zero_threshold]
        comp_split_idxs = comp_split_idxs[s > zero_threshold]

    return inp, seg, comp_split_idxs



def split_recursively(device, inp, net, seg, time_steps, n_times, zero_threshold=10):
    seg = seg[:1]
    inp = inp[None]
    orig_inp = inp
    seg = seg[None]
    orig_shape = seg.shape[2:]
    comp_split_idxs = torch.zeros((1, 2), dtype=torch.int, device=seg.device)

    tqdm_range = tqdm(range(n_times), desc=f'Splits = 1', position=1)
    for ri in tqdm_range:
        if do_component_split:
            inp, seg, comp_split_idxs = component_split(inp, seg, comp_split_idxs, zero_threshold)

        if seg.shape[0] * seg.shape[1] * seg.shape[2] > 2**20:  # (TODO: do not depend on hard-coded value)
            # batch to avoid running out of memory
            im_idxs = list(range(0, seg.shape[0], 16)) + [seg.shape[0]]
            a_s = []
            b_s = []
            for i in range(1, len(im_idxs)):
                i0, i1 = im_idxs[i - 1], im_idxs[i]
                a, b = split(device, inp[i0:i1], net, seg[i0:i1], time_steps)
                a_s.append(a)
                b_s.append(b)
            a = torch.cat(a_s, dim=0)
            b = torch.cat(b_s, dim=0)
        else:
            a, b = split(device, inp, net, seg, time_steps)

        seg = torch.cat((a, b), dim=0) > 0.5
        inp = torch.cat((inp, inp), dim=0)

        s = torch.sum(seg, dim=(1, 2, 3))
        seg = seg[s > zero_threshold]
        inp = inp[s > zero_threshold]

        if do_component_split:
            comp_split_idxs = torch.cat((comp_split_idxs, comp_split_idxs), dim=0)
            comp_split_idxs = comp_split_idxs[s > zero_threshold]

        tqdm_range.set_description(f'Splits = {seg.shape[0]}')

    if do_component_split:
        new_seg = torch.zeros((seg.shape[0], 1, orig_shape[0], orig_shape[1]))
        H, W = seg.shape[2:]
        for i in range(seg.shape[0]):
            r, c = comp_split_idxs[i]
            new_seg[i, 0, r:(r + H), c:(c + W)] = seg[i, 0]
        seg = new_seg

    return seg[:, 0]


def evaluate(model_name, dataset, recursions, steps, info, do_segmentation=None, eval_name='eval'):
    # Settings
    device = dataset[0][0].device
    Path(f'models/{model_name}/{eval_name}').mkdir(parents=True, exist_ok=True)
    if do_segmentation:
        Path(f'models/{model_name}/{eval_name}/segmentation').mkdir(exist_ok=True)
    # Load segmentation model
    in_channels = dataset[0][0].shape[0]
    seg_model = cpunet.create_net(in_channels, 1, nbase=nbase)
    seg_model.load_state_dict(torch.load(f'models/{model_name}/segmentation.pt', map_location='cpu'))
    seg_model.to(device)
    seg_model.eval()

    if do_segmentation:
        model = None
    else:
        # Load model
        in_channels = dataset[0][0].shape[0] + 1 + 2  # segmentation + a, b
        out_channels = 2  # a, b
        model = cpunet.create_net(in_channels, out_channels, nbase=nbase)
        model.load_state_dict(torch.load(f'models/{model_name}/split.pt', map_location='cpu'))
        model.to(device)
        model.eval()

    # Diffusion settings
    time_steps = torch.tensor([0] + list(np.logspace(-10, 0, (steps - 1))), device=device, dtype=torch.float)

    idx = 0
    plt.figure(figsize=(20 if do_segmentation else 10, 10))
    to_save = []
    for im, label in tqdm(dataset, position=0):
        with torch.no_grad():
            segmentation = seg_model(im[None])[0] > 0.0

        if do_segmentation:
            label_seg = torch.max(label, dim=2)[0].cpu().numpy()

            seg = segmentation[0].cpu().numpy()
            im = im.cpu().numpy()
            im = np.concatenate((np.transpose(im, [1, 2, 0]), np.zeros((im.shape[1], im.shape[2], 1))), axis=-1)
            plt.subplot(1, 3, 1)
            plt.imshow(im)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.title('Label')
            plt.imshow(label_seg)
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.title('Model')
            plt.imshow(seg)
            plt.axis('off')
            plt.savefig(f'models/{model_name}/{eval_name}/segmentation/{idx: 05d}.png')
            idx += 1

            continue

        with torch.no_grad():
            a = split_recursively(device, im, model, segmentation, time_steps, n_times=recursions)

        to_save.append({
            'im': im.cpu().numpy(),
            'label': label.cpu().numpy(),
            'prediction': a.cpu().numpy()
        })

        plt.clf()
        plt.subplot(4, 4, 1)
        im_ = np.transpose(im.cpu().numpy(), [1, 2, 0])
        if im_.shape[2] == 2:
            im_ = np.concatenate((im_, np.zeros((im_.shape[0], im_.shape[1], 1))), axis=-1)
        plt.imshow(im_)
        plt.axis('off')
        for i in range(min(15, a.shape[0])):
            plt.subplot(4, 4, i + 2)
            plt.imshow(a[i].cpu(), vmin=0, vmax=1)
            plt.axis('off')
        plt.savefig(f'models/{model_name}/{eval_name}/{idx: 05d}.png')

        idx += 1

    with open(f'models/{model_name}/{eval_name}/data.pkl', 'wb') as f:
        pickle.dump(to_save, f)

    with open(f'models/{model_name}/{eval_name}/data.npz', 'wb') as f:
        to_save_np = {}
        for i, d in enumerate(to_save):
            for k in d:
                to_save_np[f'{i:05d}_{k}'] = d[k]
        np.savez_compressed(f, **to_save_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/overlapdataset')
    parser.add_argument('--model', type=str, default='overlapdataset')
    parser.add_argument('--train-all', action='store_const', const=True)
    parser.add_argument('--train-segmentation', action='store_const', const=True)
    parser.add_argument('--train-split', action='store_const', const=True)
    parser.add_argument('--train-from-scratch', action='store_const', const=True)
    parser.add_argument('--plot', action='store_const', const=True)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval', action='store_const', const=True)
    parser.add_argument('--eval-segmentation', action='store_const', const=True)
    parser.add_argument('--eval-recursions', type=int, default=10)
    parser.add_argument('--diffusion-steps', type=int, default=100)
    parser.add_argument('--eval-name', type=str, default='eval')

    args = parser.parse_args()

    info = {'args': vars(args), 'cmd': sys.argv, 'nbase': nbase}

    if args.train_all or args.train_segmentation:
        dataset = load_dataset(args.dataset, train=True)
        train_segmentation(args.model, dataset, args.epochs, args.batch_size,
                           load=not args.train_from_scratch, plot=args.plot, info=info)

    if args.train_all or args.train_split:
        dataset = load_dataset(args.dataset, train=True)
        train_split(args.model, dataset, args.epochs, args.batch_size,
                    load=not args.train_from_scratch, plot=args.plot, info=info)

    if args.eval or args.eval_segmentation:
        dataset = load_dataset(args.dataset, train=False)
        evaluate(args.model, dataset, args.eval_recursions, args.diffusion_steps, info=info,
                 do_segmentation=args.eval_segmentation, eval_name=args.eval_name)


if __name__ == '__main__':
    main()
