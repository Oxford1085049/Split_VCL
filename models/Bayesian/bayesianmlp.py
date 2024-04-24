import torch
import torch.nn as nn
import torch.nn.functional as F

    
class BayesianNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, method = "standard", bias=False):
        super(BayesianNN, self).__init__()
        self.fc1_mean = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc1_log_var = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2_mean = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc2_log_var = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3_mean = nn.Linear(hidden_size, output_size, bias=bias)
        self.fc3_log_var = nn.Linear(hidden_size, output_size, bias=bias)
        
        self.method = method
        self.bias = bias
        
        self.init_value = -12 # = 10^-5
        self.init_mean_value = 0
        # set variance to 10^-6
        self.fc1_log_var.weight.data.fill_(self.init_value)
        self.fc2_log_var.weight.data.fill_(self.init_value)
        self.fc3_log_var.weight.data.fill_(self.init_value)
        # set all biases to 1e-6
        # self.fc1_log_var.bias.data.fill_(0)
        # self.fc2_log_var.bias.data.fill_(0)
        # self.fc3_log_var.bias.data.fill_(0)
        # self.fc2_log_var.bias.data.fill_(0)
        # self.fc3_log_var.bias.data.fill_(0)
        
        # self.fc1_mean.bias.data.fill_(0)
        # self.fc2_mean.bias.data.fill_(0)
        # self.fc3_mean.bias.data.fill_(0)
        # set the weights of the mean to 0
        self.fc1_mean.weight.data.fill_(self.init_mean_value)
        self.fc2_mean.weight.data.fill_(self.init_mean_value)
        self.fc3_mean.weight.data.fill_(self.init_mean_value)
        

    def forward(self, x):
        if self.method == "standard":
            x = torch.flatten(x, 1)
            sampled_weights1, bias_sampled_weights1 = self.sample_weights(self.fc1_mean, self.fc1_log_var)
            x = F.linear(x, sampled_weights1, bias_sampled_weights1)
            
            x = torch.relu(x)
            
            sampled_weights2, bias_sampled_weights2 = self.sample_weights(self.fc2_mean, self.fc2_log_var)
            x = F.linear(x, sampled_weights2, bias_sampled_weights2)
            
            x = torch.relu(x)
            
            sampled_weights3, bias_sampled_weights3 = self.sample_weights(self.fc3_mean, self.fc3_log_var)
            x = F.linear(x, sampled_weights3, bias_sampled_weights3)
            return x
    
        elif self.method == "reparametrization":
            x = torch.flatten(x, 1)
            x = self.apply_weights(x, self.fc1_mean, self.fc1_log_var)
            x = torch.relu(x)
            x = self.apply_weights(x, self.fc2_mean, self.fc2_log_var)
            x = torch.relu(x)
            x = self.apply_weights(x, self.fc3_mean, self.fc3_log_var)
            return x
    
    def apply_weights(self, x, mean_layer, log_var_layer):
        mean = mean_layer.weight
        log_var = log_var_layer.weight
        std = torch.exp(0.5 * log_var)
        x1 = x @ mean.T
        xsquare = x * x
        stdsquare = std * std
        delta = xsquare @ stdsquare.T
        eps = torch.randn_like(mean[:,0]).detach()
        eps = torch.relu(eps)
        
        if self.bias:
            bias_mean = mean_layer.bias
            bias_log_var = log_var_layer.bias
            bias_std = torch.exp(0.5 * bias_log_var)
            eps_bias = torch.randn_like(bias_std).detach()
            bias = bias_mean + bias_std * eps_bias
        else:
            bias = torch.zeros_like(mean[:,0])
        
        x = x1 + delta * eps + bias
        
        return x

    def sample_weights(self, mean_layer, log_var_layer):
        mean = mean_layer.weight
        log_var = log_var_layer.weight
        
        std = torch.exp(0.5 * log_var)
        
            
        eps = torch.randn_like(std).detach()  # Should be truncated ?, taking relu for now but should probably use truncated normal
        # eps = torch.relu(eps)
        
        # bias_eps = torch.relu(bias_eps)
        sampled_weights = mean + eps * std
        
        if self.bias:
            bias_mean = mean_layer.bias
            bias_log_var = log_var_layer.bias
            bias_std = torch.exp(0.5 * bias_log_var)
            bias_eps = torch.randn_like(bias_std).detach()
            # bias_eps = torch.relu(bias_eps)
            bias_sampled_weights = bias_mean + bias_eps * bias_std
        else:
            bias_sampled_weights = torch.zeros_like(mean[:,0])
        return sampled_weights, bias_sampled_weights


    def predict_mean(self, x):
        x = torch.flatten(x, 1)
        mean1 = torch.relu(self.fc1_mean(x))
        mean2 = torch.relu(self.fc2_mean(mean1))
        mean3 = self.fc3_mean(mean2)
        return mean3
    
    def get_weights(self):
        weights = []
        weights.append(self.fc1_mean.weight)
        weights.append(self.fc1_log_var.weight)
        weights.append(self.fc2_mean.weight)
        weights.append(self.fc2_log_var.weight)
        weights.append(self.fc3_mean.weight)
        weights.append(self.fc3_log_var.weight)
        if self.bias:
            weights.append(self.fc1_mean.bias)
            weights.append(self.fc1_log_var.bias)
            weights.append(self.fc2_mean.bias)
            weights.append(self.fc2_log_var.bias)
            weights.append(self.fc3_mean.bias)
            weights.append(self.fc3_log_var.bias)
        return weights
    
    def load_weights(self, weights):
        self.fc1_mean.weight = weights[0]
        self.fc1_log_var.weight = weights[1]
        self.fc2_mean.weight = weights[2]
        self.fc2_log_var.weight = weights[3]
        self.fc3_mean.weight = weights[4]
        self.fc3_log_var.weight = weights[5]
        if self.bias:
            self.fc1_mean.bias = weights[6]
            self.fc1_log_var.bias = weights[7]
            self.fc2_mean.bias = weights[8]
            self.fc2_log_var.bias = weights[9]
            self.fc3_mean.bias = weights[10]
            self.fc3_log_var.bias = weights[11]
    
    def get_variational_parameters(self):
        mu_fc1 = self.fc1_mean.weight.flatten()
        logvar_fc1 = self.fc1_log_var.weight.flatten()
        mu_fc2 = self.fc2_mean.weight.flatten()
        logvar_fc2 = self.fc2_log_var.weight.flatten()
        mu_fc3 = self.fc3_mean.weight.flatten()
        logvar_fc3 = self.fc3_log_var.weight.flatten()
        
        if self.bias:
            mu_bias_fc1 = self.fc1_mean.bias.flatten()
            logvar_bias_fc1 = self.fc1_log_var.bias.flatten()
            mu_bias_fc2 = self.fc2_mean.bias.flatten()
            logvar_bias_fc2 = self.fc2_log_var.bias.flatten()
            mu_bias_fc3 = self.fc3_mean.bias.flatten()
            logvar_bias_fc3 = self.fc3_log_var.bias.flatten()
        else:
            mu_bias_fc1 = torch.tensor([])
            logvar_bias_fc1 = torch.tensor([])
            mu_bias_fc2 = torch.tensor([])
            logvar_bias_fc2 = torch.tensor([])
            mu_bias_fc3 = torch.tensor([])
            logvar_bias_fc3 = torch.tensor([])
        
        return torch.cat([mu_fc1, mu_fc2, mu_fc3, mu_bias_fc1, mu_bias_fc2, mu_bias_fc3]), torch.cat([logvar_fc1, logvar_fc2, logvar_fc3, logvar_bias_fc1, logvar_bias_fc2, logvar_bias_fc3])
        # return torch.cat([mu_fc1, mu_fc2, mu_fc3]), torch.cat([logvar_fc1, logvar_fc2, logvar_fc3])
    
    def get_prior_parameters(self):
        means, vars = self.get_variational_parameters()
        return torch.zeros_like(means), torch.ones_like(vars)
    
    def reset_parameters(self):
        self.fc1_log_var.weight.data.fill_(self.init_value)
        self.fc2_log_var.weight.data.fill_(self.init_value)
        self.fc3_log_var.weight.data.fill_(self.init_value)
        # set all biases to 1e-6
        # self.fc1_log_var.bias.data.fill_(0)
        # self.fc2_log_var.bias.data.fill_(0)
        # self.fc3_log_var.bias.data.fill_(0)
        
        # self.fc1_mean.bias.data.fill_(0)
        # self.fc2_mean.bias.data.fill_(0)
        # self.fc3_mean.bias.data.fill_(0)
        # set the weights of the mean to 0
        self.fc1_mean.weight.data.fill_(self.init_mean_value)
        self.fc2_mean.weight.data.fill_(self.init_mean_value)
        self.fc3_mean.weight.data.fill_(self.init_mean_value)
        

if __name__ == '__main__':
    model = BayesianNN(28*28, 100, 10)
    x = torch.randn(1, 1, 28, 28)
    mean = model(x)
    print(mean.shape)
    mean = model.predict_mean(x)
    print(mean.shape)