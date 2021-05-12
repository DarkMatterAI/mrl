# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06_layers.ipynb (unless otherwise specified).

__all__ = ['Linear', 'Conv', 'Conv1d', 'Conv2d', 'Conv3d', 'LSTM', 'Conditional_LSTM', 'LSTMLM', 'Conditional_LSTMLM',
           'VAEEncoder', 'VAELSTMEncoder', 'VAEConvEncoder', 'VAELinEncoder', 'VAEDecoder', 'VAE']

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

class LSTM(nn.Module):
    def __init__(self, d_embedding, d_hidden, d_output, n_layers,
                 bidir=False, dropout=0., batch_first=True):
        super().__init__()

        self.d_embedding = d_embedding
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.n_layers = n_layers
        self.bidir = bidir
        self.n_dir = 1 if not bidir else 2
        self.batch_first = batch_first

        self.lstms = []
        self.hidden_sizes = []

        for l in range(n_layers):
            input_size = d_embedding if l==0 else d_hidden
            output_size = d_output if l==n_layers-1 else d_hidden
            output_size = output_size // self.n_dir

            hidden_size = (self.n_dir, 1, output_size)
            self.hidden_sizes.append(hidden_size)

            lstm = nn.LSTM(input_size, output_size, 1, batch_first=batch_first,
                           dropout=dropout, bidirectional=bidir)
            self.lstms.append(lstm)

        self.lstms = nn.ModuleList(self.lstms)

    def forward(self, x, hiddens=None):

        bs = x.shape[0] if self.batch_first else x.shape[1]

        if hiddens is None:
            hiddens = self.get_new_hidden(bs)
            hiddens = to_device(hiddens, x.device)

        new_hiddens = []
        for i, lstm in enumerate(self.lstms):
            x, (h,c) = lstm(x, hiddens[i])
            new_hiddens.append((h.detach(), c.detach()))

        return x, new_hiddens

    def get_new_hidden(self, bs):
        hiddens = []
        for hs in self.hidden_sizes:
            h = torch.zeros(hs).repeat(1,bs,1)
            c = torch.zeros(hs).repeat(1,bs,1)
            hiddens.append((h,c))

        return hiddens

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
                to_hidden.append(Linear(d_latent, ndir*dim*2, act=False, bn=False))

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

# Cell

class LSTMLM(nn.Module):
    def __init__(self, d_vocab, d_embedding, d_hidden, n_layers,
                 lstm_drop=0., bos_idx=0, bidir=False):
        super().__init__()

        self.embedding = nn.Embedding(d_vocab, d_embedding)
        self.lstm = LSTM(d_embedding, d_hidden, d_embedding, n_layers, bidir=bidir, dropout=lstm_drop)
        self.head = Linear(d_embedding, d_vocab, act=False, bn=False, dropout=0.)
        self.bos_idx = bos_idx

    def forward(self, x):
        x = self.embedding(x)
        x, hiddens = self.lstm(x)
        self.last_hidden = hiddens
        x = self.head(x)
        return x

    def sample(self, bs, sl, temperature=1., multinomial=True):

        preds = idxs = to_device(torch.tensor([self.bos_idx]*bs).long().unsqueeze(-1))
        lps = []

        hiddens = None

        for i in range(sl):
            x = self.embedding(idxs)
            x, hiddens = self.lstm(x, hiddens)
            x = self.head(x)

            x.div_(temperature)

            log_probs = F.log_softmax(x, -1).squeeze(1)
            probs = log_probs.detach().exp()

            if multinomial:
                idxs = torch.multinomial(probs, 1)
            else:
                idxs = x.argmax(-1)

            lps.append(torch.gather(log_probs, 1, idxs))

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

class Conditional_LSTMLM(nn.Module):
    def __init__(self, d_vocab, d_embedding, d_hidden, n_layers, mapping, d_latent,
                 lstm_drop=0., lin_drop=0., bos_idx=0, bidir=False,
                 condition_hidden=True, condition_output=False):
        super().__init__()

        self.mapping = mapping
        self.d_latent = d_latent

        self.embedding = nn.Embedding(d_vocab, d_embedding)
        self.lstm = Conditional_LSTM(d_embedding, d_hidden, d_embedding, d_latent, n_layers,
                                    condition_hidden=condition_hidden, condition_output=condition_output,
                                     bidir=bidir, dropout=lstm_drop)

        self.head = Linear(d_embedding, d_vocab, act=False, bn=False, dropout=0.)
        self.bos_idx = bos_idx

    def forward(self, x, condition):

        z = self.mapping(condition)

        x = self.embedding(x)
        x, hiddens = self.lstm(x,z)
        self.last_hidden = hiddens
        x = self.head(x)
        return x

    def sample(self, z, sl, temperature=1., multinomial=True):

        bs = z.shape[0]

        preds = idxs = to_device(torch.tensor([self.bos_idx]*bs).long().unsqueeze(-1))
        lps = []

        hiddens = None

        for i in range(sl):
            x = self.embedding(idxs)
            x, hiddens = self.lstm(x, z, hiddens)
            x = self.head(x)

            x.div_(temperature)

            log_probs = F.log_softmax(x, -1).squeeze(1)
            probs = log_probs.detach().exp()

            if multinomial:
                idxs = torch.multinomial(probs, 1)
            else:
                idxs = x.argmax(-1)

            lps.append(torch.gather(log_probs, 1, idxs))

            preds = torch.cat([preds, idxs], -1)

        return preds[:, 1:], torch.cat(lps,-1)

    def sample_no_grad(self, z, sl, temperature=1., multinomial=True):
        with torch.no_grad():
            return self.sample(z, sl, temperature=temperature, multinomial=multinomial)

    def get_lps(self, x, y, temperature=1.):
        x = self.forward(x[0], x[1])
        x.div_(temperature)

        lps = F.log_softmax(x, -1)
        lps = lps.gather(2, y.unsqueeze(-1)).squeeze(-1)

        return lps

# Cell

class VAEEncoder(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.d_latent = d_latent

    def forward(self, x):
        raise NotImplementedError

    def get_latent(self, mu, logvar, z_scale=1.):
        z = z_scale*torch.randn(mu.shape).to(mu.device)
        z = mu + z*torch.exp(0.5*logvar)
        kl_loss = 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar).sum(1).mean()
        return z, kl_loss

class VAELSTMEncoder(VAEEncoder):
    def __init__(self, d_vocab, d_embedding, d_hidden, n_layers, d_latent, dropout=0.):
        super().__init__(d_latent)

        self.embedding = nn.Embedding(d_vocab, d_embedding)
        self.lstm_encoder = LSTM(d_embedding, d_hidden, d_hidden, n_layers,
                                 bidir=True, batch_first=True, dropout=dropout)
        self.transition = nn.Linear(d_hidden*2, d_latent*2)


    def forward(self, x, z_scale=1.):
        x = self.embedding(x)
        x, hiddens = self.lstm_encoder(x)
        hidden = torch.cat(list(torch.cat(hiddens[-1], -1)), -1) # concatenate hidden/cell states of last layer

        mu, logvar = torch.chunk(self.transition(hidden), 2, dim=-1)
        z, kl_loss = self.get_latent(mu, logvar, z_scale)

        return z, kl_loss

class VAEConvEncoder(VAEEncoder):
    def __init__(self, d_vocab, d_embedding, kernel_size, n_layers, d_latent, dropout=0.):
        super().__init__(d_latent)

        self.embedding = nn.Embedding(d_vocab, d_embedding)

        convs = []
        input_size = d_embedding
        for i in range(n_layers):
            convs.append(Conv1d(input_size, input_size*2, ks=kernel_size, stride=2,
                                act=True, bn=True, dropout=dropout))
            input_size = input_size*2

        self.convs = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.transition = nn.Linear(input_size, d_latent*2)

    def forward(self, x, z_scale=1.):
        x = self.embedding(x)
        x = x.permute(0,2,1)
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)

        mu, logvar = torch.chunk(self.transition(x), 2, dim=-1)
        z, kl_loss = self.get_latent(mu, logvar, z_scale)

        return z, kl_loss

class VAELinEncoder(VAEEncoder):
    def __init__(self, d_input, n_layers, d_latent, dropout=0.):
        super().__init__(d_latent)

        lins = []
        input_size = d_input
        for i in range(n_layers):
            lins.append(Linear(input_size, input_size//2, act=True, bn=True, dropout=dropout))
            input_size = input_size//2

        self.layers = nn.Sequential(*lins)
        self.transition = nn.Linear(input_size, d_latent*2)

    def forward(self, x, z_scale=1.):
        x = self.layers(x)

        mu, logvar = torch.chunk(self.transition(x), 2, dim=-1)
        z, kl_loss = self.get_latent(mu, logvar, z_scale)

        return z, kl_loss

# Cell

class VAEDecoder(nn.Module):
    def __init__(self, d_vocab, d_embedding, d_hidden, n_layers, d_latent,
                condition_hidden=True, condition_output=True):
        super().__init__()

        self.embedding = nn.Embedding(d_vocab, d_embedding)
        self.decoder = Conditional_LSTM(d_embedding, d_hidden, d_embedding, d_latent, 3,
                                    condition_hidden=condition_hidden, condition_output=condition_output,
                                    bidir=False, batch_first=True)

        self.head = Linear(d_embedding, d_vocab, act=False, bn=False, dropout=0.)

    def forward(self, x, z, hiddens=None):
        bs, sl = x.shape
        x = self.embedding(x)

        decoded, hiddens = self.decoder(x, z, hiddens)
        output = self.head(decoded)

        return output, hiddens

# Cell

class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior=None, bos_idx=0):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        if prior is None:
            prior = Normal(torch.zeros((encoder.d_latent)), torch.ones((encoder.d_latent)))
        self.prior = prior
        self.bos_idx = bos_idx

    def forward(self, x, decoder_input=None):
        z, kl_loss = self.encoder(x)

        if decoder_input is None:
            decoder_input = x

        output, hiddens = self.decoder(decoder_input, z)
        return output, kl_loss

    def sample(self, bs, sl, z=None, temperature=1., multinomial=True):

        preds = idxs = to_device(torch.tensor([self.bos_idx]*bs).long().unsqueeze(-1))
        lps = []

        if z is None:
            z = to_device(self.prior.sample([bs]))

        hiddens = None

        for i in range(sl):
            x, hiddens = self.decoder(idxs, z, hiddens)
            x.div_(temperature)

            log_probs = F.log_softmax(x, -1).squeeze(1)
            probs = log_probs.detach().exp()

            if multinomial:
                idxs = torch.multinomial(probs, 1)
            else:
                idxs = x.argmax(-1)

            lps.append(torch.gather(log_probs, 1, idxs))

            preds = torch.cat([preds, idxs], -1)

        return preds[:, 1:], torch.cat(lps,-1)

    def sample_no_grad(self, bs, sl, z=None, temperature=1., multinomial=True):
        with torch.no_grad():
            return self.sample(bs, sl, z=z, temperature=temperature, multinomial=multinomial)

    def get_lps(self, x, y, temperature=1. z=None):

        if type(x)==list:
            x,_ = self.forward(x[0], decoder_input=x[1])
        else:
            x,_ = self.forward(x)

        x.div_(temperature)

        lps = F.log_softmax(x, -1)
        lps = lps.gather(2, y.unsqueeze(-1)).squeeze(-1)

        return lps
