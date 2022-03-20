from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    sample = np.random.normal(10,1,1000)
    sample_uni = UnivariateGaussian().fit(sample)
    mu = sample_uni.mu_
    var = sample_uni.var_
    print((mu, var))

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []
    sizes = np.linspace(10, 1000, 100).astype(int)
    # for i in range(100):
    #     print(sizes[i])
    for size in sizes:
        dif = (abs(np.mean(sample[:size+1])-mu))
        estimated_mean.append(dif)

    go.Figure([go.Scatter(x=sizes, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Absolute Distance Between The Estimated - And True Value Of The Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$|\hat\mu - \mu|$",
                               height=300)).show()


    # Question 3 - Plotting Empirical PDF of fitted model

    sample_uni_pdf = sample_uni.pdf(sample)
    go.Figure([go.Scatter(x=sample, y=sample_uni_pdf, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{PDF Of Samples ~N(10,1)}$",
                               xaxis_title="$\\text{samples}$",
                               yaxis_title="r$PDF (Density) $",
                               height=300)).show()



#
# def test_multivariate_gaussian():
#     # Question 4 - Draw samples and print fitted model
#     mu = (np.array([0,0,4,0])).transpose()
#     sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
#     sample = np.random.multivariate_normal(mu, sigma, 1000)
#     print((np.mean(sample), np.cov(sample)))
#
#
#
#     # Question 5 - Likelihood evaluation
#
#
#     # Question 6 - Maximum likelihood
#
#

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
#    test_multivariate_gaussian()
