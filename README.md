<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# artificial-neural-network
Self-implemented MLP, CNN, Dropout & BN and RNN based on NumPy.  
Homework for Artificial Neural Network, Tsinghua University, 2018-Fall.

**HW1: Multilayer Perceptron (MLP)**
* First, review the Chain rule of multivariate functions composition
* The return Gradient of ReLU layer is
```python
grad_output * (input > 0)
```
not
```python
np.maximum(grad_output, 0)
```
* Save some intermediate results during forward stage for backward
* Use matrix operation to accelerate the back propgation
```python
input = self._saved_tensor
self.grad_W = np.matmul(input.T, grad_output)
self.grad_b = np.sum(grad_output, axis=0)
return np.matmul(grad_output, self.W.T)
```
* Mini-batch Update: the Euclidean loss here divides the batch size N, so there is no need to divide N again before the weight update. They are equivalent as the credit assignment of each weights is a Linear combination of the gradients. The 'real' loss should not be related to the batch size (thus, PyTorch presumes you divide N in loss by default).


**HW1_Random: Random Synaptic Feedback**
* Reimplement the paper [Lillicrap, T. P., Cownden, D., Tweed, D. B. & Akerman, C. J. Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications*, 7, 13276 (2016)](https://www.nature.com/articles/ncomms13276) based on HW1.
* Use the architecture (784, 512, 512, 10) with ReLU.
* Test accuracy on MNIST:

|Backpropagation|Random Synaptic Feedback|
|---------------|------------------------|
|97%|95%|

* An interesting relationship between ```init_std``` of ```W``` and ```B``` and [the gradient exploding problem](https://www.bioinf.jku.at/publications/older/2604.pdf)(check section 3.1) was found. As suggested by the author, it also might be related with $e^TWBe>0$.

**HW2: Convolution Neural Network (CNN)**


**HW3: Dropout & Batch Normalization (BN)**


**HW4: Recurrent Neural Network (RNN)**
