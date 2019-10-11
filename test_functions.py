def test_multivariate_gauss():
    from net.gmm_net import GMMBlock
    import torch
    from torch.distributions.multivariate_normal import MultivariateNormal
    
    B, C, H, W = 1, 10, 1, 1
    
    x = torch.rand(B, C, H, W)
    
    mu = torch.rand(B, C, H, W)
    sigma = torch.rand(B, 1, H, W)
    
    actual = torch.log(GMMBlock.multivariate_Gaussian(x, mu, sigma))
        
    gauss = MultivariateNormal(mu[0, :, 0, 0], torch.eye(C) * torch.pow(sigma[0, 0, 0, 0], 2))
    expected = gauss.log_prob(x[0, :, 0, 0])
    
    print("actual", actual[0, 0, 0, 0])
    print("expected", expected)
    
def test_jaccard_loss():
    pass

if __name__ == "__main__":
    test_multivariate_gauss()