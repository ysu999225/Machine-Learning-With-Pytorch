import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(26, 200),
            nn.ReLU(),
            nn.Linear(200,400),
            nn.ReLU(),
            nn.Linear(400,200),
            nn.ReLU(),
            nn.Linear(200,2)
        )

    def forward(self, x, sigma):

        # NeRF-Style positional embedding of x. This helps the function
        # learn high frequency functions more easily. 
        # new_x is 26 dimensional. 

        x_cos_emb = torch.exp2(torch.linspace(0, 5, 6))
        x_sin_emb = torch.exp2(torch.linspace(0, 5, 6))

        y_cos_emb = torch.exp2(torch.linspace(0, 5, 6))
        y_sin_emb = torch.exp2(torch.linspace(0, 5, 6))

        x_cos_emb = torch.cos(torch.outer(x[:,0], x_cos_emb) * torch.pi)
        x_sin_emb = torch.sin(torch.outer(x[:,0], x_sin_emb) * torch.pi)

        y_cos_emb = torch.cos(torch.outer(x[:,1], y_cos_emb) * torch.pi)
        y_sin_emb = torch.sin(torch.outer(x[:,1], y_sin_emb) * torch.pi)

        new_x = torch.hstack([x_cos_emb, x_sin_emb, y_cos_emb, y_sin_emb, x])
        return self.net(new_x) / sigma # Scale network by sigma following NCSNv2

class ScoreMatching():

    def __init__(self):
        self.scorenet = DiffusionModel()
    
    '''
    Utility function to visualize the score function learned by this model. 
    '''
    def plot_score(self):
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xv, yv = np.meshgrid(x,y)
        arrow_starts = np.stack([xv.flatten(), yv.flatten()], axis=1).astype(np.float32)

        arrow_ends = (self.scorenet(torch.from_numpy(arrow_starts), 0.03)).detach().numpy()

        plt.quiver(arrow_starts[:,1], arrow_starts[:,0], arrow_ends[:,1], arrow_ends[:,0])
        plt.title("score function")
        plt.show()

    '''
    Generate initial random noise to denoise.
    
    A gaussian distribution that roughly fits the ([0,1], [0,1]) range in which the points lie. 
    '''
    def initial_random_samples(self):
        return torch.randn((2000,2))/3 + 0.5

    '''
    Generate geometrically distributed sigmas (sigma_0, sigma_1, ..., sigma_n) starting from smallest going to largest.  
    '''
    def generate_noise_levels(sigma_smallest, sigma_largest, num_levels):
        return torch.tensor(np.exp(np.linspace(np.log(sigma_smallest), np.log(sigma_largest),num_levels))).float()

    '''
    Denoise x_mod using Langevin Dynamics, with:
        - noise_levels being a (L,) dimension tensor containing sigmas
        - iterations specifying the number of iterations to perform
        - lr specifying the learning rate for langevin dynamics 
    '''
    @torch.no_grad()
    def langevin_dynamics_sample(self, x_mod, noise_levels, iterations, lr):
        # iterate over the noise levels from high to low
        for i in noise_levels:
            for n in range(iterations):
                # self.scorenet = DiffusionModel()
                score = self.scorenet(x_mod,i)
                x_mod = x_mod + lr * score
                # gaussian noise add to the samples
                noise_scale = torch.sqrt(torch.tensor(lr * 2))
                noise = torch.randn_like(x_mod) * noise_scale
                x_mod = x_mod + noise
                # range [0,1]
                x_mod = torch.clamp(x_mod,0,1)

        return x_mod



    '''
    Calculate denoising score matching loss based on samples. 
        - samples is (2000, 2) dimension tensor containing the dataset
        - noise_levels is a (L) dimension tensor containing your chosen noise levels
    '''
    def denoising_loss(self, samples, noise_levels):
        
        sigma_chosen = noise_levels[torch.randint(len(noise_levels), (samples.shape[0],))]

        #TODO: YOUR CODE HERE
        # add gaussian noise to samples
        noised_samples = samples + torch.randn_like(samples) * sigma_chosen.unsqueeze(1)
        # get the prediction of score
        prediction = self.scorenet(noised_samples, sigma_chosen.unsqueeze(1))
        # true 
        true = -(noised_samples - samples) / sigma_chosen.unsqueeze(1)**2
        loss = torch.mean((prediction-true) ** 2)
        return loss


    '''
    Fit self.scorenet to the provided data argument at provided noise_levels. 
    '''
    def train(self, data, noise_levels, learning_rate, iterations):
        optimizer = torch.optim.Adam(self.scorenet.parameters(),lr=learning_rate)
        #TODO: YOUR CODE HERE
        for i in range(iterations):
            # make the gradients to be 0
            optimizer.zero_grad()
            # loss and back propagation
            loss = self.denoising_loss(data, noise_levels)
            loss.backward()
            # update the parameters
            optimizer.step()
            

def main():

    #TODO: SET HYPERPARAMETERS HERE
    retrain = True # Uncomment to disable training from scratch
    #(0.1,0.01,10)
    # I find the more narrow in noise_levels, the better performance
    noise_levels = ScoreMatching.generate_noise_levels(0.1, 0.01, 10)
    training_lr = 1e-4
    training_iterations = 10000

    sampling_iterations = 2000
    sampling_lr = 1e-4

    #Visualize Dataset
    coords = torch.from_numpy(np.load('cs446.npy')).float()
    plt.scatter(coords[:,1], coords[:,0])
    plt.title("true samples")
    plt.show()


    #Train or load the model
    sm = None
    if(retrain):
        sm = ScoreMatching()
        sm.train(coords, noise_levels, training_lr, training_iterations)
        torch.save(sm, "model.pth")
    else: 
        sm = torch.load("model.pth")
    
    #Visualize score function
    sm.plot_score()

    #Generate and plot samples 
    samples = sm.langevin_dynamics_sample(sm.initial_random_samples(), noise_levels, sampling_iterations, sampling_lr).numpy()

    plt.scatter(samples[:,1], samples[:,0])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("generated samples")
    plt.show()


if __name__ == '__main__':
    main()
