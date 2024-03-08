import sys
import argparse
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import matplotlib.image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hw5_utils import *




# The "encoder" model q(z|x)
class Encoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Encoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units
        
        self.fc1 = nn.Linear(data_dimension, hidden_units)
        self.fc2_mu = nn.Linear(hidden_units, latent_dimension)
        self.fc2_sigma = nn.Linear(hidden_units, latent_dimension)

    def forward(self, x):
        # Input: x input image [batch_size x data_dimension]
        # Output: parameters of a diagonal gaussian 
        #   mean : [batch_size x latent_dimension]
        #   variance : [batch_size x latent_dimension]

        hidden = torch.tanh(self.fc1(x))
        mu = self.fc2_mu(hidden)
        log_sigma_square = self.fc2_sigma(hidden)
        sigma_square = torch.exp(log_sigma_square)  
        return mu, sigma_square


# "decoder" Model p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Decoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units

        # TODO: deine the parameters of the decoder
        # fc1: a fully connected layer with 500 hidden units. 
        #map the latent space back to the data space p(x|z)
        # inverse order as above q(z|x)
        self.fc1 = nn.Linear(latent_dimension, 500)
        self.tanh = nn.Tanh()
        # fc2: a fully connected layer with 500 hidden units. 
        self.fc2 = nn.Linear(500, data_dimension)
        self.sigmoid = nn.Sigmoid()
        #activate function
        #tanh after the first layer and sigmoid after the second layer
        
    def forward(self, z):
        # input
        #   z: latent codes sampled from the encoder [batch_size x latent_dimension]
        # output 
        #   p: a tensor of the same size as the image indicating the probability of every pixel being 1 [batch_size x data_dimension]

        # TODO: implement the decoder here. The decoder is a multi-layer perceptron with two hidden layers. 
        # The first layer is followed by a tanh non-linearity and the second layer by a sigmoid.
        layer1 = torch.tanh(self.fc1(z))
        p = torch.sigmoid(self.fc2(layer1))
        return p


# VAE model
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dimension = args.latent_dimension
        self.hidden_units =  args.hidden_units
        self.data_dimension = args.data_dimension
        self.resume_training = args.resume_training
        self.batch_size = args.batch_size
        self.num_epoches = args.num_epoches
        self.e_path = args.e_path
        self.d_path = args.d_path

        # load and pre-process the data
        N_data, self.train_images, self.train_labels, test_images, test_labels = load_mnist()

        # Instantiate the encoder and decoder models 
        self.encoder = Encoder(self.latent_dimension, self.hidden_units, self.data_dimension)
        self.decoder = Decoder(self.latent_dimension, self.hidden_units, self.data_dimension)

        # Load the trained model parameters
        if self.resume_training:
            self.encoder.load_state_dict(torch.load(self.e_path))
            self.decoder.load_state_dict(torch.load(self.d_path))

    # Sample from Diagonal Gaussian z~N(μ,σ^2 I) 
    @staticmethod
    def sample_diagonal_gaussian(mu, sigma_square):
        # Inputs:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   sample: from a diagonal gaussian with mean mu and variance sigma_square [batch_size x latent_dimension]

        # TODO: Implement the reparameterization trick and return the sample z [batch_size x latent_dimension]
        
        # get the dimension size of sigma_square
        #Generates random samples from a standard normal distribution (mean 0, variance 1) with the same shape as sigma_square
        #torch.randn_like(sigma_square)
        size = sigma_square.size()
        # Implement the reparameterization trick randomly
        trick = torch.randn(size)
        sample = mu + torch.sqrt(sigma_square) * trick
        return sample

    # Sampler from Bernoulli
    @staticmethod
    def sample_Bernoulli(p):
        # Input: 
        #   p: the probability of pixels labeled 1 [batch_size x data_dimension]
        # Output:
        #   x: pixels'labels [batch_size x data_dimension]

        # TODO: Implement a sampler from a Bernoulli distribution
        # just apply the Bernoulli distribution
        x = torch.bernoulli(p)
        return x


    # Compute Log-pdf of z under Diagonal Gaussian N(z|μ,σ^2 I)
    @staticmethod
    def logpdf_diagonal_gaussian(z, mu, sigma_square):
        # Input:
        #   z: sample [batch_size x latent_dimension]
        #   mu: mean of the gaussian distribution [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian distribution [batch_size x latent_dimension]
        # Output:
        #    logprob: log-probability of a diagomnal gaussian [batch_size]
        
        # TODO: implement the logpdf of a gaussian with mean mu and variance sigma_square*I
        logprob = -0.5 * (torch.log(2 * torch.pi * sigma_square)+(z - mu)**2 / sigma_square)
        #sum log prob across the latent dimensions.
        logprob = torch.sum(logprob,dim = 1)
        return logprob

    # Compute log-pdf of x under Bernoulli 
    @staticmethod
    def logpdf_bernoulli(x, p):
        # Input:
        #   x: samples [batch_size x data_dimension]
        #   p: the probability of the x being labeled 1 (p is the output of the decoder) [batch_size x data_dimension]
        # Output:
        #   logprob: log-probability of a bernoulli distribution [batch_size]
        logprob = x * torch.log(p) + (1 - x) * torch.log(1 - p)
        #same as above sum log prob across the latent dimensions.
        logprob = torch.sum(logprob,dim = 1)
        # TODO: implement the log likelihood of a bernoulli distribution p(x)
        return logprob
    
    # Sample z ~ q(z|x)
    def sample_z(self, mu, sigma_square):
        # input:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   zs: samples from q(z|x) [batch_size x latent_dimension] 
        zs = self.sample_diagonal_gaussian(mu, sigma_square)
        return zs 


    # Variational Objective
    def elbo_loss(self, sampled_z, mu, sigma_square, x, p):
        # Inputs
        #   sampled_z: samples z from the encoder [batch_size x latent_dimension]
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        #   x: data samples [batch_size x data_dimension]
        #   p: the probability of a pixel being labeled 1 [batch_size x data_dimension]
        # Output
        #   elbo: the ELBO loss (scalar)

        # log_q(z|x) logprobability of z under approximate posterior N(μ,σ)
        log_q = self.logpdf_diagonal_gaussian(sampled_z, mu, sigma_square)
        
        # log_p_z(z) log probability of z under prior
        z_mu = torch.FloatTensor([0]*self.latent_dimension)
        z_sigma = torch.FloatTensor([1]*self.latent_dimension)
        log_p_z = self.logpdf_diagonal_gaussian(sampled_z, z_mu, z_sigma)

        # log_p(x|z) - conditional probability of data given latents.
        log_p = self.logpdf_bernoulli(x, p)
        
        # TODO: implement the ELBO loss using log_q, log_p_z and log_p
        DKL_divergence = log_q - log_p_z
        elbo = log_p - DKL_divergence
        # average the batch
        elbo = torch.mean(elbo)  
        return elbo


    def train(self):
        
        # Set-up ADAM optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        adam_optimizer = optim.Adam(params)

        # Train for ~200 epochs 
        num_batches = int(np.ceil(len(self.train_images) / self.batch_size))
        num_iters = self.num_epoches * num_batches
        
        for i in range(num_iters):
            x_minibatch = self.train_images[batch_indices(i, num_batches, self.batch_size),:]
            adam_optimizer.zero_grad()

            mu, sigma_square = self.encoder(x_minibatch)
            zs = self.sample_z(mu, sigma_square)
            p = self.decoder(zs)
            elbo = self.elbo_loss(zs, mu, sigma_square, x_minibatch, p)
            total_loss = -elbo
            total_loss.backward()
            adam_optimizer.step()

            if i%100 == 0:
                print("Epoch: " + str(i//num_batches) + ", Iter: " + str(i) + ", ELBO:" + str(elbo.item()))

        # Save Optimized Model Parameters
        torch.save(self.encoder.state_dict(), self.e_path)
        torch.save(self.decoder.state_dict(), self.d_path)


    # Generate digits using the VAE
    def visualize_data_space(self):
        # TODO: Sample 10 z from prior

        mu = torch.zeros(10,self.latent_dimension)
        sigma = torch.ones(10,self.latent_dimension)
        images = [np.zeros((28,28)) for _ in range(20)]

        # print(images.shape)

        # TODO: For each z, plot p(x|z)

        for i in range(10):
            # for each pixel determine the probability for each label whether this is 0 or 1
            z = self.sample_diagonal_gaussian(mu[i], sigma[i])
            p = self.decoder(z).detach()
            arr = array_to_image(p)
            if i == 0:
                plt.plot(arr)
                #plt.show()
            images[i] = arr
            # TODO: Sample x from p(x|z)

            # to determine the label besed on the probablity
            # this is like a classification problem
            p = self.sample_Bernoulli(p)

            arr = array_to_image(p)
            if i == 0:
                fig = plt.figure()
                plt.imshow(arr)
            images[i + 10] = arr

        # TODO: Concatenate plots into a figure (use the function concat_images)
        result = concat_images(images, 10, 2)

        # TODO: Save the generated figure and include it in your report
        fig = plt.figure()
        plt.imshow(result)
        # plt.savefig('.png')  # Save as a PNG file
        plt.show()
        
    # Produce a scatter plot in the latent space, where each point in the plot will be the mean vector 
    # for the distribution $q(z|x)$ given by the encoder. Further, we will colour each point in the plot 
    # by the class label for the input data. Each point in the plot is colored by the class label for 
    # the input data.
    # The latent space should have learned to distinguish between elements from different classes, even though 
    # we never provided class labels to the model!
    def visualize_latent_space(self):
        
        # TODO: Encode the training data self.train_images
        mu, sigma_square = self.encoder(self.train_images)

        # TODO: Take the mean vector of each encoding
        data = np.array(mu.detach())

        # TODO: Plot these mean vectors in the latent space with a scatter
        # Colour each point depending on the class label 
        X = data[:,0]
        Y =data[:,1]
        tmp = np.array(self.train_labels.detach())
        label = np.argmax(tmp, axis=1)
        plt.scatter(X, Y, c=label)
        plt.xlabel('Z1')
        plt.ylabel('Z2')
        plt.title('2D Latent Scatter Plot ')
        plt.colorbar(label='Label')
        plt.savefig('2DLatent.png')  # Save as a PNG file
        plt.show()


        # TODO: Save the generated figure and include it in your report

    # Function which gives linear interpolation z_α between za and zb
    @staticmethod
    def interpolate_mu(mua, mub, alpha = 0.5):
        return alpha*mua + (1-alpha)*mub


    # A common technique to assess latent representations is to interpolate between two points.
    # Here we will encode 3 pairs of data points with different classes.
    # Then we will linearly interpolate between the mean vectors of their encodings. 
    # We will plot the generative distributions along the linear interpolation.
    def visualize_inter_class_interpolation(self):
        # TODO: Sample 3 pairs of data with different classes
        set = {0}
        idx = []
        count = 0
        while(True):
            if len(idx) == 3:
                break
            if self.train_labels[count] in set:
                continue
            else:
                set.add(self.train_labels[count])
                idx.append(count)
                count += 1


        # TODO: Encode the data in each pair, and take the mean vectors
        mu_list = []
        for i in idx:
            image = self.train_images[i]
            mu, _ = self.encoder(image)
            mu_list.append(np.array(mu.detach()))

        # TODO: Linearly interpolate between these mean vectors (Use the function interpolate_mu)
        z_alp =  np.empty((3, 11), dtype=object)

        for i in range(len(mu_list)):
            mua = mu_list[i]
            for j in range(i + 1, len(mu_list)):
                mub = mu_list[j]
                for t in range(11):
                    cur = t * 0.1
                    z_alp[i + j - 1][t] = self.interpolate_mu(mua,mub, cur)

        # TODO: Along the interpolation, plot the distributions p(x|z_α)
        
        images = [np.zeros((28,28)) for _ in range(z_alp.shape[0] * z_alp.shape[1])]
        for i in range(z_alp.shape[0]):
            for j in range(z_alp.shape[1]):
                cur_mu = torch.tensor(z_alp[i][j])
                cur_sigma = torch.ones(1, self.latent_dimension)
                cur_z = self.sample_z(cur_mu, cur_sigma)
                prob = self.decoder(cur_z).detach()
                arr = array_to_image(prob)
                images[i + j * int(z_alp.shape[0])] = arr

        # Concatenate these plots into one figure
        result = concat_images(images, 3, 11)

        fig = plt.figure()
        plt.imshow(result)
        # plt.savefig('.png')  # Save as a PNG file
        #plt.show()

      

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--e_path', type=str, default="./e_params.pkl", help='Path to the encoder parameters.')
    parser.add_argument('--d_path', type=str, default="./d_params.pkl", help='Path to the decoder parameters.')
    parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units of the encoder and decoder models.')
    parser.add_argument('--latent_dimension', type=int, default='2', help='Dimensionality of the latent space.')
    parser.add_argument('--data_dimension', type=int, default='784', help='Dimensionality of the data space.')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_epoches', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')

    args = parser.parse_args()
    return args


def main():
    print("hello")
    # read the function arguments
    args = parse_args()

    # set the random seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # train the model 
    vae = VAE(args)
    vae.train()

    # visualize the latent space
    vae.visualize_data_space()
    vae.visualize_latent_space()
    vae.visualize_inter_class_interpolation()


if __name__ == "__main__":
    main()