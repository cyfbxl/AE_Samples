
import torch
import torch.nn as nn
from torch.nn import functional as F
class Base_VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.dataset_name =='cifar':
            in_channels = 3
            self.in_channels = 3
            
        if args.dataset_name =='minst':
            in_channels = 1
            self.in_channels = 1
        
        hidden_dims = [16, 32, 16, 8]
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, args.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, args.latent_dim)
        self.decoder_input = nn.Linear(args.latent_dim, hidden_dims[-1]*4)
        
        # self.classify = nn.Sequential(
        #     nn.Linear(args.latent_dim, 32),
        #     nn.Dropout(),
        #     nn.Linear(32, 10)
        # )
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(args.latent_dim,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )
        
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        self.decoder = nn.Sequential(*modules)
        
        
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())
        
        
    # def reparameterize(self, mu, logvar):
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu
    
    def reparameterize(self, mu, logvar):
        std = (logvar.clamp(-50, 50).exp() + 1e-8) ** 0.5
        eps = torch.randn_like(logvar)
        return eps * std + mu
       
    def encode(self, input):

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # print(result.shape)
        result = result.view(-1, 8, 2, 2)
        # print(result.shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def forward(self, input, label=None):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        if label is not None:
            classify = self.classify(z)
            out = self.decode(z)
            return  [out, input, mu, log_var, classify, label]
        else:
            classify = self.classify(z)
            out = self.decode(z)
            return  [out, input, mu, log_var, classify]
        
    def get_z(self,input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z
    
    def loss_function(self, args):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        if len(args) == 6:
            recons = args[0]
            input = args[1]
            mu = args[2]
            log_var = args[3]
            classify = args[4]
            label = args[5]
            classify_loss = F.cross_entropy(classify, label)
        else:
            recons = args[0]
            input = args[1]
            mu = args[2]
            log_var = args[3]
            
        kld_weight = 0.1 # Account for the minibatch samples from the dataset
        recons_weight = 100.0
        classify_weight = 0.5
        # F.binary_cross_entropy(recon_x, x, reduction='sum')
        recons_loss = F.mse_loss(recons, input,reduction='sum') # 得用sum，要不然不行
        # F.binary_cross_entropy(recons, input, reduction='sum') 这个也行
        
        # F.mse_loss(recons, input)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        if len(args) == 6:
            loss = recons_weight * recons_loss + kld_weight * kld_loss + classify_weight * classify_loss
            # loss = classify_loss
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach(), 'classify_loss':classify_loss.detach()}
        else:
            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}