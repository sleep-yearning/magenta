For installation and usage, please see the magenta root readme file.
Below are some more background informations from the original coconet implementation.

## Coconet: Counterpoint by Convolution

Machine learning models of music typically break up the
task of composition into a chronological process, composing
a piece of music in a single pass from beginning to
end. On the contrary, human composers write music in
a nonlinear fashion, scribbling motifs here and there, often
revisiting choices previously made. In order to better
approximate this process, we train a convolutional neural
network to complete partial musical scores, and explore the
use of blocked Gibbs sampling as an analogue to rewriting.
Neither the model nor the generative procedure are tied to
a particular causal direction of composition.
Our model is an instance of orderless NADE (Uria 2014),
which allows more direct ancestral sampling. However,
we find that Gibbs sampling greatly improves sample quality,
which we demonstrate to be due to some conditional
distributions being poorly modeled. Moreover, we show
that even the cheap approximate blocked Gibbs procedure
from (Yao, 2014) yields better samples than ancestral sampling,
based on both log-likelihood and human evaluation.

Paper can be found at https://ismir2017.smcnus.org/wp-content/uploads/2017/10/187_Paper.pdf.
Huang, C. Z. A., Cooijmans, T., Roberts, A., Courville, A., & Eck, D. (2016). Counterpoint by Convolution. International Society of Music Information Retrieval (ISMIR).

## References:

Uria, B., Murray, I., & Larochelle, H. (2014, January). A deep and tractable density estimator. In International Conference on Machine Learning (pp. 467-475).

Yao, L., Ozair, S., Cho, K., & Bengio, Y. (2014, September). On the equivalence between deep nade and generative stochastic networks. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 322-336). Springer, Berlin, Heidelberg.
