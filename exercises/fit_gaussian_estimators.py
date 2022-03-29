import numpy.random

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    random_samples = numpy.random.normal(10, 1, 1000)
    gaus = UnivariateGaussian()
    gaus.fit(random_samples)
    print(gaus.mu_, gaus.var_)

    # Question 2 - Empirically showing sample mean is consistent
    exp_actual_dist = []
    for i in range(10,1000,10):
        #creating new gaussians with different means
        new_gaus = UnivariateGaussian()
        new_gaus.fit(random_samples[:i])
        exp_actual_dist.append([i,abs(new_gaus.mu_ - 10)])
    exp_actual_dist = np.array(exp_actual_dist)
    fig1 = px.line(x=exp_actual_dist[ :,0], y=exp_actual_dist[ :, 1],
                      title="Absolute value of difference between actual and estimated expectency as a function of sample size",
                      labels={'x': 'Number of samples ', 'y': 'Absolute value of distance'})
    fig1.update_layout(title_x=0.5)
    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model

    fig2 = px.scatter(x= random_samples, y=gaus.pdf(random_samples),
                      title="samples and their pdf values according to the estimated gaussian",
                      labels={'x': 'Sample value',
                              'y': 'Pdf of the sample'})
    fig2.update_layout(title_x=0.5)
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    sigma = np.array([[1,0.2,0,0.5]
                         ,[0.2,2,0,0]
                         ,[0,0,1,0],
                      [0.5,0,0,1]
                      ])
    random_samples = np.random.multivariate_normal(mu, sigma, 1000)
    gaus = MultivariateGaussian()
    gaus.fit(random_samples)
    print(gaus.mu_)
    print(gaus.cov_)

    # Question 5 - Likelihood evaluation
    f1, f3 = np.linspace(-10, 10, 200), np.linspace(-10, 10, 200)
    likelihoods = np.zeros((f1.size,f3.size))
    #insert value of likelihoods in a numpy array
    for i,v1 in enumerate(f1):
        for j, v3 in enumerate(f1):
            new_mu = np.array([v1, 0, v3, 0])
            likelihoods[i,j] = MultivariateGaussian.log_likelihood(new_mu, sigma, random_samples)

    trace = go.Heatmap(
        x=f1,
        y=f3,
        z=likelihoods,
        type='heatmap',
        colorscale='Viridis'

    )
    data = [trace]
    fig = go.Figure(data=data, layout=go.Layout(title="Log likelihood of different models"))
    fig.update_layout(xaxis_title="f3 values", yaxis_title="f1 values")
    fig.show()

    # Question 6 - Maximum likelihood
    linear_space = np.linspace(-10, 10, 200)
    indices = np.unravel_index(likelihoods.argmax(), likelihoods.shape)
    f1_max_value = np.round(linear_space[indices[0]], 3)
    f3_max_value = np.round(linear_space[indices[1]], 3)
    print(f1_max_value, f3_max_value)



if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    # test_multivariate_gaussian()
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    mean = np.array([0, 0, 4, 0])
    S = np.random.multivariate_normal(mean, cov, 1000)
    y = MultivariateGaussian()
    y.fit(S)
    print(y.log_likelihood(mean, cov, S))

