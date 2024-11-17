import torch

def perchannel_conv(x, filters):
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda:0")
ones = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda:0")
sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device="cuda:0")
lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device="cuda:0")
gaus = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device="cuda:0")


def perception(x, mask_n=0):

    filters = torch.stack([sobel_x, sobel_x.T, lap])
    if mask_n != 0:
        n = x.shape[1]
        padd = torch.zeros((x.shape[0], 3 * mask_n, x.shape[2], x.shape[3]), device="cuda:0")
        obs = perchannel_conv(x[:, 0:n - mask_n], filters)
        return torch.cat((x, obs, padd), dim=1)
    else:
        obs = perchannel_conv(x, filters)
        return torch.cat((x,obs), dim = 1 )

def masked_perception(x, mask_n=0):

    filters = torch.stack([sobel_x, sobel_x.T, lap])
    mask = torch.zeros_like(x)
    mask[:,0:x.shape[1]- mask_n,...] = 1
    x_masked = x*mask


    obs = perchannel_conv(x_masked,filters)
    return torch.cat((x,obs), dim = 1 )


def reduced_perception(x, mask_n=0):

    filters = torch.stack([sobel_x, sobel_x.T, lap])
    x_redu = x[:,0:x.shape[1]-mask_n]
    obs = perchannel_conv(x_redu,filters)
    return torch.cat((x,obs), dim = 1 )

class DummyVCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, mask_n=0):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(4 * chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.mask_n = mask_n

    def forward(self, x, update_rate=0.5):
        y = perception(x, self.mask_n)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp  = torch.nn.functional.pad(x[:, None, 3, ...],pad = [1,1,1,1] ,mode= "circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0,).cuda() > 0.1
        # Perform update
        x = x + y * update_mask * pre_life_mask
        return x

class MaskedCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, mask_n=0):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(4 * chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.mask_n = mask_n

    def forward(self, x, update_rate=0.5):
        y = masked_perception(x, self.mask_n)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp  = torch.nn.functional.pad(x[:, None, 3, ...],pad = [1,1,1,1] ,mode= "circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0).cuda() > 0.1
        # Perform update
        x = x + y * update_mask * pre_life_mask
        return x


class ReducedCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, mask_n=0):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn + 3*(chn-  mask_n), hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.mask_n = mask_n

    def forward(self, x, update_rate=0.5):
        y = reduced_perception(x, self.mask_n)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp  = torch.nn.functional.pad(x[:, None, 3, ...],pad = [1,1,1,1] ,mode= "circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0,).cuda() > 0.1
        # Perform update
        x = x + y * update_mask * pre_life_mask
        return x


class GeneCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size=3):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn + 3 * (chn), hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn - gene_size, 1, bias=False)
        self.w2.weight.data.zero_()
        self.gene_size = gene_size

    def forward(self, x, update_rate=0.5):
        gene = x[:, -self.gene_size:, ...]
        y = reduced_perception(x, 0)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0, ).cuda() > 0.1
        x = x[:, :x.shape[1] - self.gene_size, ...] + y * update_mask * pre_life_mask
        x = torch.cat((x, gene), dim=1)
        return x


class GenePropCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size=3):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(4*chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n,  gene_size, 1, bias=False)
        self.w2.weight.data.zero_()
        self.gene_size = gene_size

    def forward(self, x, update_rate=0.5):
        gene = x[:, -self.gene_size:, ...]
        y = reduced_perception(x, 0)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0, ).cuda() > 0.1
        gene = gene + y  * update_mask* pre_life_mask
        x = x[:, :x.shape[1] - self.gene_size, ...]
        x = torch.cat((x, gene), dim=1)
        return x


def gradnorm_perception(x):
  grad = perchannel_conv(x, torch.stack([sobel_x, sobel_x.T]))
  gx, gy = grad[:, ::2], grad[:, 1::2]
  state_lap = perchannel_conv(x, torch.stack([ident, lap]))
  return torch.cat([ state_lap, (gx*gx+gy*gy+1e-8).sqrt()], 1)


class IsoCA(torch.nn.Module):
  def __init__(self, chn=12, hidden_n=128,  gene_size=3):
    super().__init__()
    self.chn = chn

    # Determine the number of perceived channels
    perc_n = gradnorm_perception(torch.zeros([1, chn, 8, 8], device="cuda:0")).shape[1]

    self.gene_size = gene_size
    self.w1 = torch.nn.Conv2d(perc_n, hidden_n, 1)
    self.w2 = torch.nn.Conv2d(hidden_n, chn- gene_size, 1, bias=False)
    self.w2.weight.data.zero_()



  def forward(self, x, update_rate=0.5):

    gene = x[:, -self.gene_size:, ...]
    y = gradnorm_perception(x)
    y = self.w1(y)
    y = self.w2(torch.nn.functional.leaky_relu(y))
    b, c, h, w = y.shape
    update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
    pre_life_mask = torch.nn.functional.max_pool2d(x[:,None,3,...], 3, 1, 1).cuda() > 0.1
    x = x[:, :x.shape[1] - self.gene_size, ...] + y * update_mask * pre_life_mask
    x = torch.cat((x, gene), dim=1)

    return x



class IsoGenePropCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size=3):
        super().__init__()
        self.chn = chn
        perc_n = gradnorm_perception(torch.zeros([1, chn, 8, 8], device="cuda:0")).shape[1]
        self.w1 = torch.nn.Conv2d(perc_n, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n,  gene_size, 1, bias=False)
        self.w2.weight.data.zero_()
        self.gene_size = gene_size

    def forward(self, x, update_rate=0.5):
        gene = x[:, -self.gene_size:, ...]
        y = gradnorm_perception(x)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0, ).cuda() > 0.1
        gene = gene + y  * update_mask* pre_life_mask
        x = x[:, :x.shape[1] - self.gene_size, ...]
        x = torch.cat((x, gene), dim=1)
        return x