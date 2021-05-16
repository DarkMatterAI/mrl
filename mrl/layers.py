# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_layers.ipynb (unless otherwise specified).

__all__ = ['Linear', 'Conv', 'Conv1d', 'Conv2d', 'Conv3d', 'SphericalDistribution', 'Prior', 'NormalPrior',
           'SphericalPrior', 'Conditional_LSTM', 'LSTM', 'Conditional_LSTM_Block', 'LSTM_Block', 'LSTM_LM', 'Encoder',
           'LSTM_Encoder', 'MLP_Encoder', 'Conv_Encoder', 'VAE_Transition', 'Norm_Transition', 'PT_Transition',
           'Encoder_Decoder', 'VAE', 'LSTM_VAE', 'Conv_VAE', 'MLP_VAE', 'Conditional_LSTM_LM']

# Cell
from .imports import *
from .torch_imports import *
from .torch_core import *

# Cell

class Linear(nn.Module):
    def __init__(self, d_in, d_out, act=True, bn=False, dropout=0., **lin_kwargs):
        super().__init__()

        layers = [nn.Linear(d_in, d_out, **lin_kwargs)]

        if bn:
            layers.append(nn.BatchNorm1d(d_out))

        if act:
            layers.append(nn.ReLU())

        if dropout>0.:
            layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Conv(nn.Module):
    def __init__(self, d_in, d_out, ks=3, stride=1, padding=None, ndim=2,
                 act=True, bn=False, dropout=0., **conv_kwargs):
        super().__init__()

        if padding is None:
            padding = (ks-1)//2

        if ndim==1:
            conv_func = nn.Conv1d
            bn_func = nn.BatchNorm1d
        elif ndim==2:
            conv_func = nn.Conv2d
            bn_func = nn.BatchNorm2d
        else:
            conv_func = nn.Conv3d
            bn_func = nn.BatchNorm3d

        layers = [conv_func(d_in, d_out, ks, stride, padding=padding, **conv_kwargs)]

        if bn:
            layers.append(bn_func(d_out))

        if act:
            layers.append(nn.ReLU())

        if dropout>0.:
            layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Conv1d(Conv):
    def __init__(self, d_in, d_out, ks=3, stride=1, padding=None,
                 act=True, bn=False, dropout=0., **conv_kwargs):
        super().__init__(d_in, d_out, ks, stride, padding, ndim=1,
                 act=act, bn=bn, dropout=dropout, **conv_kwargs)

class Conv2d(Conv):
    def __init__(self, d_in, d_out, ks=3, stride=1, padding=None,
                 act=True, bn=False, dropout=0., **conv_kwargs):
        super().__init__(d_in, d_out, ks, stride, padding, ndim=2,
                 act=act, bn=bn, dropout=dropout, **conv_kwargs)

class Conv3d(Conv):
    def __init__(self, d_in, d_out, ks=3, stride=1, padding=None,
                 act=True, bn=False, dropout=0., **conv_kwargs):
        super().__init__(d_in, d_out, ks, stride, padding, ndim=3,
                 act=act, bn=bn, dropout=dropout, **conv_kwargs)

# Cell

class SphericalDistribution(torch.distributions.Distribution):
    def __init__(self, loc, scale, validate_args=False):
        super().__init__(loc.shape[0], validate_args=validate_args)
        self.dim = loc.shape[0]
        self.loc = loc
        self.scale = scale
        self.dist = Normal(self.loc, self.scale)

    def sample(self, n):
        s = self.dist.sample(n)
        s = F.normalize(s, p=2, dim=-1)
        return s

    def rsample(self, n):
        s = self.dist.rsample(n)
        s = F.normalize(s, p=2, dim=-1)
        return s

    def __repr__(self):
        return f'Spherical(loc: {self.loc.size()}, scale: {self.scale.size()})'

class Prior(nn.Module):
    def __init__(self):
        super().__init__()

    def get_dist(self):
        raise NotImplementedError

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, n):
        if type(n)==int:
            n = [n]
        return self.get_dist().sample(n)

    def rsample(self, n):
        if type(n)==int:
            n = [n]
        return self.get_dist().rsample(n)


class NormalPrior(Prior):
    def __init__(self, loc, log_scale, trainable=True):
        super().__init__()
        if trainable:
            loc = nn.Parameter(loc)
            log_scale = nn.Parameter(log_scale)
        self.loc = loc
        self.log_scale = log_scale
        self.trainable = trainable

    def get_dist(self):
        return Normal(self.loc, self.log_scale.exp())

    def log_prob(self, x):
        var = self.log_scale.exp().pow(2)
        return -((x - self.loc) ** 2) / (2 * var) - self.log_scale - math.log(math.sqrt(2 * math.pi))

class SphericalPrior(NormalPrior):
    def __init__(self, loc, log_scale, trainable=True):
        super().__init__(loc, log_scale, trainable)

    def get_dist(self):
        return SphericalDistribution(self.loc, self.log_scale.exp())

# Cell

class Conditional_LSTM(nn.Module):
    def __init__(self, d_embedding, d_hidden, d_output, d_latent, n_layers,
                 condition_hidden=True, condition_output=True,
                 bidir=False, dropout=0., batch_first=True):
        super().__init__()

        self.d_embedding = d_embedding
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.n_layers = n_layers
        self.bidir = bidir
        self.n_dir = 1 if not bidir else 2
        self.batch_first = batch_first
        self.condition_hidden = condition_hidden
        self.condition_output = condition_output

        self.lstms = []
        self.hidden_sizes = []

        for l in range(n_layers):
            if l==0:
                input_size = d_embedding if not self.condition_output else d_embedding+d_latent
            else:
                input_size = d_hidden

            output_size = d_output if l==n_layers-1 else d_hidden
            output_size = output_size // self.n_dir

            hidden_size = (self.n_dir, 1, output_size)
            self.hidden_sizes.append(hidden_size)

            lstm = nn.LSTM(input_size, output_size, 1, batch_first=batch_first,
                           dropout=dropout, bidirectional=bidir)
            self.lstms.append(lstm)

        self.lstms = nn.ModuleList(self.lstms)

        if self.condition_hidden:
            to_hidden = []
            for size in self.hidden_sizes:
                ndir, _, dim = size
                to_hidden.append(nn.Linear(d_latent, ndir*dim*2))

            self.to_hidden = nn.ModuleList(to_hidden)

    def forward(self, x, z, hiddens=None):

        bs = x.shape[0] if self.batch_first else x.shape[1]
        sl = x.shape[1] if self.batch_first else x.shape[0]

        if self.condition_output:
            if self.batch_first:
                z_ = z.unsqueeze(1).repeat(1,sl,1)
            else:
                z_ = z.unsqueeze(0).repeat(sl,1,1)

            x = torch.cat([x, z_], -1)

        if hiddens is None:
            if self.condition_hidden:
                hiddens = self.latent_to_hidden(z)

            else:
                hiddens = self.get_new_hidden(bs)

            hiddens = to_device(hiddens, x.device)

        new_hiddens = []
        for i, lstm in enumerate(self.lstms):
            x, (h,c) = lstm(x, hiddens[i])
            new_hiddens.append((h.detach(), c.detach()))

        return x, new_hiddens

    def latent_to_hidden(self, z):
        hiddens = []
        for layer in self.to_hidden:
            h = layer(z)
            h,c = torch.chunk(h, 2, dim=-1)
            bs, _ = h.shape
            h = h.contiguous().reshape(bs, self.n_dir, -1).permute(1,0,2)
            c = c.contiguous().reshape(bs, self.n_dir, -1).permute(1,0,2)
            hiddens.append((h,c))

        return hiddens

    def get_new_hidden(self, bs):
        hiddens = []
        for hs in self.hidden_sizes:
            h = torch.zeros(hs).repeat(1,bs,1)
            c = torch.zeros(hs).repeat(1,bs,1)
            hiddens.append((h,c))

        return hiddens

    def mixup_hiddens(self, hiddens):
        new_hiddens = []
        for item in hiddens:
            h,c = item
            shuffle = to_device(torch.randperm(h.shape[1]))
            h = h[:,shuffle]
            c = c[:,shuffle]
            new_hiddens.append((h,c))
        return new_hiddens

class LSTM(Conditional_LSTM):
    def __init__(self, d_embedding, d_hidden, d_output, n_layers,
                 bidir=False, dropout=0., batch_first=True):
        super().__init__(d_embedding, d_hidden, d_output, 0, n_layers,
                 condition_hidden=False, condition_output=False,
                 bidir=bidir, dropout=dropout, batch_first=batch_first)

    def forward(self, x, hiddens=None):

        x, new_hiddens = super().forward(x, None, hiddens)

        return x, new_hiddens

# Cell

class Conditional_LSTM_Block(nn.Module):
    def __init__(self, d_vocab, d_embedding, d_hidden, d_output, d_latent, n_layers,
                 lstm_drop=0., lin_drop=0., bidir=False,
                 condition_hidden=True, condition_output=False):
        super().__init__()

        self.embedding = nn.Embedding(d_vocab, d_embedding)
        self.lstm = Conditional_LSTM(d_embedding, d_hidden, d_output, d_latent, n_layers,
                                    condition_hidden=condition_hidden, condition_output=condition_output,
                                     bidir=bidir, dropout=lstm_drop)

        self.head = Linear(d_output, d_vocab, act=False, bn=False, dropout=lin_drop)

    def forward(self, x, z, hiddens=None):
        x = self.embedding(x)
        x, hiddens = self.lstm(x, z, hiddens)
        x = self.head(x)

        return x, hiddens

class LSTM_Block(nn.Module):
    def __init__(self, d_vocab, d_embedding, d_hidden, d_output, n_layers,
                 lstm_drop=0., lin_drop=0., bidir=False):
        super().__init__()

        self.embedding = nn.Embedding(d_vocab, d_embedding)
        self.lstm = LSTM(d_embedding, d_hidden, d_output, n_layers,
                                     bidir=bidir, dropout=lstm_drop)

        self.head = nn.Linear(d_output, d_vocab)
        self.head_drop = nn.Dropout(lin_drop)

    def forward(self, x, hiddens=None):
        x = self.embedding(x)
        x, hiddens = self.lstm(x, hiddens)
        x = self.head_drop(self.head(x))

        return x, hiddens


# Cell

class LSTM_LM(nn.Module):
    def __init__(self, d_vocab, d_embedding, d_hidden, n_layers,
                 lstm_drop=0., lin_drop=0., bos_idx=0, bidir=False, tie_weights=False):
        super().__init__()

        self.block = LSTM_Block(d_vocab, d_embedding, d_hidden, d_embedding, n_layers,
                                lstm_drop=lstm_drop, lin_drop=lin_drop, bidir=bidir)
        self.bos_idx = bos_idx

        if tie_weights:
            self.block.embedding.weight = self.block.head.weight

    def forward(self, x, hiddens=None):
        x, hiddens = self.block(x, hiddens)
        return x

    def sample(self, bs, sl, temperature=1., multinomial=True):

        preds = idxs = to_device(torch.tensor([self.bos_idx]*bs).long().unsqueeze(-1))
        lps = []

        hiddens = None

        for i in range(sl):
            x, hiddens = self.block(idxs, hiddens)
            x.div_(temperature)

            idxs, lp = x_to_preds(x, multinomial=multinomial)

            lps.append(lp)
            preds = torch.cat([preds, idxs], -1)

        return preds[:, 1:], torch.cat(lps,-1)

    def sample_no_grad(self, bs, sl, temperature=1., multinomial=True):
        with torch.no_grad():
            return self.sample(bs, sl, temperature=temperature, multinomial=multinomial)

    def get_lps(self, x, y, temperature=1.):
        x = self.forward(x)
        x.div_(temperature)

        lps = F.log_softmax(x, -1)
        lps = lps.gather(2, y.unsqueeze(-1)).squeeze(-1)

        return lps

# Cell
class Encoder(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.d_latent = d_latent

class LSTM_Encoder(Encoder):
    def __init__(self, d_vocab, d_embedding, d_hidden, n_layers, d_latent, dropout=0.):
        super().__init__(d_latent)
        self.embedding = nn.Embedding(d_vocab, d_embedding)
        self.lstm = LSTM(d_embedding, d_hidden, d_hidden, n_layers,
                                 bidir=True, batch_first=True, dropout=dropout)
        self.head = nn.Linear(d_hidden*2, d_latent)

    def forward(self, x):
        x = self.embedding(x)
        x, hiddens = self.lstm(x)
        hidden = torch.cat(list(torch.cat(hiddens[-1], -1)), -1) # concatenate hidden/cell states of last layer
        latent = self.head(hidden)
        return latent

class MLP_Encoder(Encoder):
    def __init__(self, d_in, dims, d_latent, dropouts):
        super().__init__(d_latent)

        dims = [d_in]+dims

        acts = [True]*(len(dims)-1)
        bns = [True]*(len(dims)-1)
        layers = [Linear(d_in, d_out, act=a, bn=b, dropout=p)
                 for d_in, d_out, a, b, p in zip(dims[:-1], dims[1:], acts, bns, dropouts)]
        layers.append(nn.Linear(dims[-1], d_latent))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Conv_Encoder(Encoder):
    def __init__(self, d_vocab, d_embedding, d_latent, filters, kernel_sizes, strides, dropouts):
        super().__init__(d_latent)

        self.embedding = nn.Embedding(d_vocab, d_embedding)

        filters = [d_embedding] + filters

        convs = [Conv1d(filters[i], filters[i+1], ks=kernel_sizes[i],
                        stride=strides[i], act=True, bn=True, dropout=dropouts[i])
                    for i in range(len(filters)-1)]

        self.convs = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(filters[-1], d_latent)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0,2,1)
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)
        x = self.head(x)
        return x

# Cell

class VAE_Transition(nn.Module):
    def __init__(self, d_latent):
        super().__init__()

        self.d_latent = d_latent
        self.transition = nn.Linear(d_latent, d_latent*2)

    def forward(self, x, z_scale=1.):
        mu, logvar = self.get_stats(x)
        z = z_scale*torch.randn(mu.shape).to(mu.device)
        z = mu + z*torch.exp(0.5*logvar)
        kl_loss = 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar).sum(1).mean()
        return z, kl_loss

    def get_stats(self, x):
        mu, logvar = torch.chunk(self.transition(x), 2, dim=-1)
        return mu, logvar

class Norm_Transition(nn.Module):
    def __init__(self, d_latent, p=2):
        super().__init__()
        self.d_latent = d_latent
        self.p = p

    def forward(self, x):
        x = F.normalize(x, p=self.p, dim=-1)
        return x

class PT_Transition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# Cell

class Encoder_Decoder(nn.Module):
    def __init__(self, encoder, decoder, transition=None, prior=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if transition is None:
            transition = PT_Transition()

        self.transition = transition

        if prior is None:
            prior = NormalPrior(torch.zeros((encoder.d_latent)), torch.zeros((encoder.d_latent)),
                                trainable=False)

        self.prior = prior

    def forward(self, x, decoder_input=None):
        if decoder_input is None:
            decoder_input = x

        z = self.encoder(x)
        z = self.transition(x)
        output = self.decoder(decoder_input, z)
        return output

    def set_prior(self, prior):
        self.prior = prior

# Cell

class VAE(Encoder_Decoder):
    def __init__(self, encoder, decoder, prior=None, bos_idx=0):
        transition = VAE_Transition(encoder.d_latent)
        super().__init__(encoder, decoder, transition, prior)

        self.bos_idx = bos_idx

    def forward(self, x, decoder_input=None):

        z = self.encoder(x)
        z, kl_loss = self.transition(z)

        if decoder_input is None:
            decoder_input = x

        output, hiddens = self.decoder(decoder_input, z)
        return output, kl_loss


    def sample(self, bs, sl, z=None, temperature=1., multinomial=True):

        preds = idxs = to_device(torch.tensor([self.bos_idx]*bs).long().unsqueeze(-1))
        lps = []

        if z is None:
            z = to_device(self.prior.rsample([bs]))

        hiddens = None

        for i in range(sl):
            x, hiddens = self.decoder(idxs, z, hiddens)
            x.div_(temperature)

            idxs, lp = x_to_preds(x, multinomial=multinomial)

            lps.append(lp)
            preds = torch.cat([preds, idxs], -1)

        return preds[:, 1:], torch.cat(lps,-1)

    def sample_no_grad(self, bs, sl, z=None, temperature=1., multinomial=True):
        with torch.no_grad():
            return self.sample(bs, sl, z=z, temperature=temperature, multinomial=multinomial)

    def get_lps(self, x, y, temperature=1., z=None):

        if type(x)==list:
            z,_ = self.transition(self.encoder(x[0]))
            x,_ = self.decoder(x[1], z)
        else:
            z,_ = self.transition(self.encoder(x))
            x,_ = self.decoder(x,z)

        x.div_(temperature)

        lps = F.log_softmax(x, -1)
        lps = lps.gather(2, y.unsqueeze(-1)).squeeze(-1)

        if self.prior.trainable:
            prior_lps = self.prior.log_prob(z).mean(-1, keepdim=True)
            prior_lps = torch.zeros(prior_lps.shape).float() + prior_lps - prior_lps.detach()
            lps += prior_lps

        return lps

    def set_prior_from_stats(self, mu, logvar, trainable=False):
        mu = mu.detach()
        logvar = logvar.detach()
        self.prior = NormalPrior(mu, logvar, trainable)

    def set_prior_from_latent(self, z, trainable=False):
        mu, logvar = self.transition.get_stats(z)
        self.set_prior_from_stats(mu, logvar, trainable)

    def set_prior_from_encoder(self, x, trainable=False):
        assert x.shape[0]==1, "Must set prior from a single input"
        z = self.encoder(x)
        z = z.squeeze(0)
        self.set_prior_from_latent(z, trainable)

# Cell

class LSTM_VAE(VAE):
    def __init__(self, d_vocab, d_embedding, d_hidden, n_layers, d_latent,
                enc_drop=0., dec_drop=0., condition_hidden=True, condition_output=True,
                prior=None, bos_idx=0):

        encoder = LSTM_Encoder(d_vocab, d_embedding, d_hidden,
                               n_layers, d_latent, dropout=enc_drop)

        decoder = Conditional_LSTM_Block(d_vocab, d_embedding, d_hidden, d_embedding,
                                d_latent, n_layers, lstm_drop=dec_drop, lin_drop=dec_drop,
                                condition_hidden=condition_hidden, condition_output=condition_output)

        super().__init__(encoder, decoder, prior, bos_idx)

# Cell

class Conv_VAE(VAE):
    def __init__(self, d_vocab, d_embedding, conv_filters, kernel_sizes, strides, conv_drops,
                 d_hidden, n_layers, d_latent, dec_drop=0.,
                 condition_hidden=True, condition_output=True,
                 prior=None, bos_idx=0):

        encoder = Conv_Encoder(d_vocab, d_embedding, d_latent,
                               conv_filters, kernel_sizes, strides, conv_drops)

        decoder = Conditional_LSTM_Block(d_vocab, d_embedding, d_hidden, d_embedding,
                                d_latent, n_layers, lstm_drop=dec_drop, lin_drop=dec_drop,
                                condition_hidden=condition_hidden, condition_output=condition_output)

        super().__init__(encoder, decoder, prior, bos_idx)

# Cell

class MLP_VAE(VAE):
    def __init__(self, d_vocab, d_embedding, encoder_d_in, encoder_dims, encoder_drops,
                 d_hidden, n_layers, d_latent, dec_drop=0.,
                 condition_hidden=True, condition_output=True,
                 prior=None, bos_idx=0):

        encoder = MLP_Encoder(encoder_d_in, encoder_dims, d_latent, encoder_drops)

        decoder = Conditional_LSTM_Block(d_vocab, d_embedding, d_hidden, d_embedding,
                                d_latent, n_layers, lstm_drop=dec_drop, lin_drop=dec_drop,
                                condition_hidden=condition_hidden, condition_output=condition_output)

        super().__init__(encoder, decoder, prior, bos_idx)



# Cell

class Conditional_LSTM_LM(Encoder_Decoder):
    def __init__(self, encoder, d_vocab, d_embedding, d_hidden, d_latent, n_layers,
                 lstm_drop=0., lin_drop=0., bidir=False,
                 condition_hidden=True, condition_output=False, bos_idx=0, prior=None):

        transition = Norm_Transition(d_latent)

        decoder = Conditional_LSTM_Block(d_vocab, d_embedding, d_hidden, d_embedding,
                                d_latent, n_layers, lstm_drop=lstm_drop, lin_drop=lin_drop,
                                condition_hidden=condition_hidden, condition_output=condition_output)

        if prior is None:
            prior = SphericalPrior(torch.zeros((encoder.d_latent)), torch.zeros((encoder.d_latent)),
                                trainable=False)

        super().__init__(encoder, decoder, transition, prior)

        self.bos_idx = bos_idx

    def forward(self, x, condition, hiddens=None):
        z = self.encoder(condition)
        z = self.transition(z)
        x, hiddens = self.decoder(x, z, hiddens)
        return x


    def sample(self, bs, sl, z=None, temperature=1., multinomial=True):

        if z is None:
            if self.prior is not None:
                z = to_device(self.prior.rsample([bs]))
            else:
                z = to_device(torch.randn((bs, self.encoder.d_latent)))
                z = self.transition(z)
        else:
            bs = z.shape[0]

        preds = idxs = to_device(torch.tensor([self.bos_idx]*bs).long().unsqueeze(-1))
        lps = []

        hiddens = self.decoder.lstm.latent_to_hidden(z)

        for i in range(sl):

            x, hiddens = self.decoder(idxs,z,hiddens)
            x.div_(temperature)

            idxs, lp = x_to_preds(x, multinomial=multinomial)

            lps.append(lp)
            preds = torch.cat([preds, idxs], -1)

        return preds[:, 1:], torch.cat(lps,-1)

    def sample_no_grad(self, bs, sl, z=None, temperature=1., multinomial=True):
        with torch.no_grad():
            return self.sample(bs, sl, z=z, temperature=temperature, multinomial=multinomial)

    def get_lps(self, x, y, temperature=1.):
        x, c = x
        z = self.transition(self.encoder(c))
        x,_ = self.decoder(x, z)

        x.div_(temperature)

        lps = F.log_softmax(x, -1)
        lps = lps.gather(2, y.unsqueeze(-1)).squeeze(-1)

        if self.prior.trainable:
            prior_lps = self.prior.log_prob(z).mean(-1, keepdim=True)
            prior_lps = torch.zeros(prior_lps.shape).float() + prior_lps - prior_lps.detach()
            lps += prior_lps

        return lps

    def set_prior_from_latent(self, z, logvar, trainable=False):
        z = z.detach()
        logvar = logvar.detach()
        self.prior = SphericalPrior(z, logvar, trainable)

    def set_prior_from_encoder(self, condition, logvar, trainable=False):
        assert condition.shape[0]==1
        z = self.transition(self.encoder(condition))
        z = z.squeeze(0)
        self.set_prior_from_latent(z, logvar, trainable)