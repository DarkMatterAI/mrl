# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/11_generative_models.vae.ipynb (unless otherwise specified).

__all__ = ['VAE_Transition', 'VAE', 'LSTM_VAE', 'Conv_VAE', 'MLP_VAE', 'VAELoss']

# Cell

from ..imports import *
from ..torch_imports import *
from ..torch_core import *
from ..layers import *
from .generative_base import *

# Cell

class VAE_Transition(nn.Module):
    def __init__(self, d_latent):
        super().__init__()

        self.d_latent = d_latent
        self.transition = nn.Linear(d_latent, d_latent*2)

    def forward(self, x, z_scale=None):
        if z_scale is None:
            z_scale = 1.
        mu, logvar = self.get_stats(x)
        z = z_scale*torch.randn(mu.shape).to(mu.device)
        z = mu + z*torch.exp(0.5*logvar)
        kl_loss = 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar).sum(1).mean()
        return z, kl_loss

    def get_stats(self, x):
        mu, logvar = torch.chunk(self.transition(x), 2, dim=-1)
        return mu, logvar

# Cell

class VAE(GenerativeModel):
    def __init__(self, encoder, decoder, prior=None, bos_idx=0, transition=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        if prior is None:
            prior = NormalPrior(torch.zeros((encoder.d_latent)), torch.zeros((encoder.d_latent)),
                                trainable=False)

        self.prior = prior

        if transition is None:
            transition = VAE_Transition(encoder.d_latent)

        self.transition = transition

        self.bos_idx = bos_idx
        self.z_scale = 1.

    def forward(self, decoder_input, encoder_input=None, hiddens=None):
        if encoder_input is None:
            encoder_input = decoder_input

        z = self.encoder(encoder_input)
        z, kl_loss = self.transition(z, self.z_scale)
        output, hiddens, encoded = self.decoder(decoder_input, z, hiddens)
        return output, kl_loss

    def unpack_x(self, x):
        if isinstance(x, (list, tuple)):
            decoder_input, encoder_input = x
        else:
            encoder_input = x
            decoder_input = x

        return decoder_input, encoder_input

    def x_to_latent(self, x):
        decoder_input, encoder_input = self.unpack_x(x)

        z = self.encoder(encoder_input)
        z,_ = self.transition(z, self.z_scale)
        return z

    def sample(self, bs, sl, z=None, temperature=1., multinomial=True, z_scale=1.):

        current_device = next(self.parameters()).device

        start_idx = torch.tensor([self.bos_idx]*bs).long().unsqueeze(-1)

        preds = idxs = to_device(start_idx, device=current_device)

        lps = []

        if z is None:
            if self.prior is not None:
                z = to_device(self.prior.rsample([bs]), device=current_device)
            else:
                prior = Normal(torch.zeros((self.encoder.d_latent)),
                               torch.ones((self.encoder.d_latent)))
                z = to_device(self.prior.rsample([bs]), device=current_device)
        else:
            bs = z.shape[0]

        hiddens = None

        for i in range(sl):
            x, hiddens, encoded = self.decoder(idxs, z, hiddens)
            x.div_(temperature)

            idxs, lp = x_to_preds(x, multinomial=multinomial)

            lps.append(lp)
            preds = torch.cat([preds, idxs], -1)

        return preds[:, 1:], torch.cat(lps,-1)

    def sample_no_grad(self, bs, sl, z=None, temperature=1., multinomial=True):
        with torch.no_grad():
            return self.sample(bs, sl, z=z, temperature=temperature, multinomial=multinomial)

    def get_rl_tensors(self, x, y, temperature=1., latent=None):

        decoder_input, encoder_input = self.unpack_x(x)

        if latent is None:
            latent = self.encoder(encoder_input)
            latent,_ = self.transition(latent)

        output, hiddens, encoded = self.decoder(decoder_input, latent)

        output.div_(temperature)
        lps = F.log_softmax(output, -1)

        if self.prior.trainable:
            prior_lps = self.prior.log_prob(z)
            prior_lps = prior_lps.mean(-1).unsqueeze(-1).unsqueeze(-1)
            pass_through = torch.zeros(prior_lps.shape).float().to(prior_lps.device)
            pass_through = pass_through + prior_lps - prior_lps.detach() # add to gradient chain
            lps = lps + pass_through

        lps_gathered = gather_lps(lps, y)
        return output, lps, lps_gathered, encoded

    def set_prior_from_stats(self, mu, logvar, trainable=False):
        mu = mu.detach()
        logvar = logvar.detach()
        self.prior = NormalPrior(mu, logvar, trainable)

    def set_prior_from_latent(self, z, trainable=False):
        mu, logvar = self.transition.get_stats(z)
        self.set_prior_from_stats(mu, logvar, trainable)

    def set_prior_from_encoder(self, x, trainable=False):
        decoder_input, encoder_input = self.unpack_x(x)
        assert encoder_input.shape[0]==1, "Must set prior from a single input"
        z = self.encoder(encoder_input)
        z = z.squeeze(0)
        self.set_prior_from_latent(z, trainable)

# Cell

class LSTM_VAE(VAE):
    def __init__(self, d_vocab, d_embedding, d_hidden, n_layers, d_latent,
                input_dropout=0., lstm_dropout=0.,
                condition_hidden=True, condition_output=True,
                prior=None, bos_idx=0, transition=None):

        encoder = LSTM_Encoder(
                                d_vocab,
                                d_embedding,
                                d_hidden,
                                n_layers,
                                d_latent,
                                input_dropout=input_dropout,
                                lstm_dropout=lstm_dropout
                            )

        decoder = Conditional_LSTM_Block(
                                d_vocab,
                                d_embedding,
                                d_hidden,
                                d_embedding,
                                d_latent,
                                n_layers,
                                input_dropout=input_dropout,
                                lstm_dropout=lstm_dropout,
                                bidir=False,
                                condition_hidden=condition_hidden,
                                condition_output=condition_output,
                            )


        super().__init__(encoder, decoder, prior, bos_idx, transition)

# Cell

class Conv_VAE(VAE):
    def __init__(self, d_vocab, d_embedding,
                 conv_filters, kernel_sizes, strides, conv_drops,
                 d_hidden, n_layers, d_latent,
                 input_dropout=0., lstm_dropout=0.,
                 condition_hidden=True, condition_output=True,
                 prior=None, bos_idx=0, transition=None):

        encoder = Conv_Encoder(
                                d_vocab,
                                d_embedding,
                                d_latent,
                                conv_filters,
                                kernel_sizes,
                                strides,
                                conv_drops,
                            )

        decoder = Conditional_LSTM_Block(
                                d_vocab,
                                d_embedding,
                                d_hidden,
                                d_embedding,
                                d_latent,
                                n_layers,
                                input_dropout=input_dropout,
                                lstm_dropout=lstm_dropout,
                                bidir=False,
                                condition_hidden=condition_hidden,
                                condition_output=condition_output,
                            )

        super().__init__(encoder, decoder, prior, bos_idx, transition)

# Cell

class MLP_VAE(VAE):
    def __init__(self, d_vocab, d_embedding, encoder_d_in, encoder_dims, encoder_drops,
                 d_hidden, n_layers, d_latent,
                 input_dropout=0., lstm_dropout=0.,
                 condition_hidden=True, condition_output=True,
                 prior=None, bos_idx=0, transition=None):


        encoder = MLP_Encoder(encoder_d_in, encoder_dims, d_latent, encoder_drops)

        decoder = Conditional_LSTM_Block(
                                d_vocab,
                                d_embedding,
                                d_hidden,
                                d_embedding,
                                d_latent,
                                n_layers,
                                input_dropout=input_dropout,
                                lstm_dropout=lstm_dropout,
                                bidir=False,
                                condition_hidden=condition_hidden,
                                condition_output=condition_output,
                            )

        super().__init__(encoder, decoder, prior, bos_idx, transition)



# Cell

class VAELoss():
    def __init__(self, weight=1.):
        self.ce = CrossEntropy()
        self.weight = weight

    def __call__(self, inputs, targs):
        output, kl_loss = inputs
        return self.ce(output, targs) + self.weight*kl_loss