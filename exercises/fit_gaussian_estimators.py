from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


#
def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    sample = np.random.normal(10, 1, 1000)
    sample_uni = UnivariateGaussian().fit(sample)
    mu = sample_uni.mu_
    print(mu)
    var = sample_uni.var_
    print((mu, var))

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []
    sizes = np.linspace(10, 1000, 100).astype(int)
    # for i in range(100):
    #     print(sizes[i])
    for size in sizes:
        dif = (abs((UnivariateGaussian().fit(sample[:size])).mu_ - mu))
        estimated_mean.append(dif)
    go.Figure([go.Scatter(x=sizes, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Absolute Distance Between The Estimated - And True Value Of The Expectation As Function Of Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$|\hat\mu - \mu|$",
                  height=300)).show()
    #
    #
    #     # Question 3 - Plotting Empirical PDF of fitted model

    sample_uni_pdf = sample_uni.pdf(sample)
    go.Figure([go.Scatter(x=sample, y=sample_uni_pdf, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{PDF Of Samples ~N(10,1)}$",
                               xaxis_title="$\\text{samples}$",
                               yaxis_title="r$PDF (Density) $",
                               height=300)).show()


# new_s = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
#               -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
# print(UnivariateGaussian().log_likelihood(10, 1, new_s))


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    sample = np.random.multivariate_normal(mean=mu, cov=sigma, size=1000)
    sample_multi = MultivariateGaussian().fit(sample)
    print(sample_multi.mu_)
    print(sample_multi.cov_)
    # print(sample)

    # Question 5 - Likelihood evaluation
    res = {}
    log_likelihood = []
    samples_f1 = np.linspace(-10, 10, 200)
    samples_f3 = np.linspace(-10, 10, 200)
    for f1 in samples_f1:
        row = []
        for f3 in samples_f3:
            log_lh = sample_multi.log_likelihood(np.array([f1, 0, f3, 0]), sample_multi.cov_, sample)
            row.append(log_lh)
            res[(f1, f3)] = log_lh
        log_likelihood.append(row)

    go.Figure(go.Heatmap(x=samples_f1, y=samples_f3, z=log_likelihood),
              layout=go.Layout(title=r"$\text{Log Likelihood As Functions Of F1, F3 (features)}$",
                               xaxis_title="$f1$",
                               yaxis_title="r$f3 $")).show()

    # Question 6 - Maximum likelihood
    print("%.3f, %.3f" % max(res, key=res.get))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
