import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, IterableDataset
from constants import DEFORMATOR_TYPE_DICT

class ShiftedGANSamplesDataset(IterableDataset):
    def __init__(self, G, dim, scales=(-10.0, 10.0), batch_size=32, deformator=None):
        super(ShiftedGANSamplesDataset, self).__init__()
        self.G = G
        self.deformator = deformator
        self.scales = scales
        self.batch = batch_size
        self.dim = dim

    @torch.no_grad()
    def __iter__(self):
        z = torch.randn([self.batch, self.G.dim_z] if type(self.G.dim_z) == int \
                            else [self.batch] + self.G.dim_z, device='cuda')

        shifts_neg = self.scales[0] * torch.rand([self.batch], device='cuda')
        shifts_pos = self.scales[1] * torch.rand([self.batch], device='cuda')

        shifts = torch.zeros([2 * self.batch, self.G.dim_z], device='cuda')
        shifts[:self.batch, self.dim] = shifts_neg
        shifts[self.batch:, self.dim] = shifts_pos
        if self.deformator is not None:
            shifts = self.deformator(shifts)

        imgs = self.G.gen_shifted(z.repeat(2, 1), shifts)
        yield (imgs[:self.batch], imgs[self.batch:]), (shifts_neg, shifts_pos)


class PseudoLabelDataset(IterableDataset):
    def __init__(self, G, dim, r=6.0, batch_size=32, deformator=None, size=None):
        super(PseudoLabelDataset, self).__init__()
        if deformator is not None:
            assert dim < deformator.input_dim, 'dim [{}] < deformator.input_dim [{}]'.format(
                dim, deformator.input_dim)

        self.G = G
        self.deformator = deformator
        self.r = r
        self.batch = batch_size
        self.dim = dim
        self.size = size

    @torch.no_grad()
    def __iter__(self):
        while True:
            latent_shape = [self.batch, self.G.dim_z] if type(self.G.dim_z) == int else \
                [self.batch] + self.G.dim_z


            signs = torch.randint(0, 2, [self.batch], device='cuda')

            deformator_input_shape = [self.batch, self.G.dim_shift]
            if self.deformator is not None and self.deformator.type is not DEFORMATOR_TYPE_DICT['id']:
                deformator_input_shape[-1] = self.deformator.input_dim
            shifts = torch.zeros(deformator_input_shape, device='cuda')
            shifts[:, self.dim] = self.r * (2.0 * signs - 1.0)
            if self.deformator is not None:
                shifts = self.deformator(shifts)
                shifts = shifts.view([-1] + self.G.dim_z)

            with torch.no_grad():
                # Image Generation
                while True:
                    # z = torch.randn(latent_shape, device='cuda')
                    z = torch.normal(mean=torch.zeros(latent_shape), std=torch.ones(latent_shape) * 0.6).cuda()
                    imgs = self.G.nvp_shifted(z, shifts, reverse=True)  # .clamp_(0, 1)
                    if torch.sum(torch.isinf(imgs)) == 0:
                        imgs = torch.clamp(imgs, min=0, max=1)
                        break
                    print("Inf values in generated images!   Repeat image generation")
                    del imgs
                    torch.cuda.empty_cache()

            if self.size is not None:
                imgs = F.interpolate(imgs, self.size)

            yield imgs, signs


class ModelLabeledDataset(Dataset):
    def __init__(self, ds, model):
        super(ModelLabeledDataset, self).__init__()
        self.ds = ds
        self.model = model

    def __len__(self):
        return len(self.ds)

    @torch.no_grad()
    def __getitem__(self, item):
        img = self.ds[item].cuda()
        logits = self.model(img.unsqueeze(0))
        label = torch.argmax(logits, dim=1).squeeze()

        return img, label
