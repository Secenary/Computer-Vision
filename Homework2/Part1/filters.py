import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    ### YOUR CODE HERE
    for n in range(Hi):
        for m in range(Wi):
            for k in range(Hk):
                for l in range(Wk):
                    if n - k + Hk // 2 in range(Hi) and m - l + Wk // 2 in range(Wi):
                        out[n, m] += kernel[k, l] * image[n - k + Hk // 2, m - l + Wk // 2]
    ### END YOUR CODE
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H + pad_height * 2, W + pad_width * 2))

    ### YOUR CODE HERE
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image; 
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    zero_padded_image = zero_pad(image, Hk // 2, Wk // 2)
    fliped_kernel = np.flip(kernel)
    ### YOUR CODE HERE
    for n in range(Hi):
        for m in range(Wi):
            out[n, m] = np.sum(zero_padded_image[n:n + Hk, m:m + Wk] * fliped_kernel)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_zero_mean = g - np.mean(g)
    out = cross_correlation(f, g_zero_mean)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    
    mu_g = np.mean(g)
    sigma_g = np.std(g)
    if sigma_g == 0:
        sigma_g = 1e-10  # Avoid division by zero
    
    g_centered = g - mu_g
    
    pad_height = Hg // 2
    pad_width = Wg // 2
    padded_f = zero_pad(f, pad_height, pad_width)
    
    out = np.zeros((Hf, Wf))
    
    for n in range(Hf):
        for m in range(Wf):
            window = padded_f[n:n+Hg, m:m+Wg]
            mu_window = np.mean(window)
            window_centered = window - mu_window
            sigma_window = np.std(window)
            
            if sigma_window == 0:
                out[n, m] = 0
            else:
                correlation = np.sum(window_centered * g_centered)
                out[n, m] = correlation / (sigma_window * sigma_g)
    ### END YOUR CODE

    return out
