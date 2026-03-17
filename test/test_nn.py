# TODO: import dependencies and write unit tests below

import numpy as np
from nn.nn import NeuralNetwork
from nn.preprocess import one_hot_encode_seqs, sample_seqs


def _build_test_network(loss_function: str = "bce") -> NeuralNetwork:
    """
    A helper function for creating a simple network for unit tests.
    """
    nn_arch = [
        {"input_dim": 2, "output_dim": 2, "activation": "relu"},
        {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"},
    ]
    return NeuralNetwork(nn_arch, lr=0.1, seed=0, batch_size=2, epochs=2, loss_function=loss_function)


def test_single_forward():
    """
    Check that a single layer computes both linear output and ReLU activation.
    """
    # init a dummy network and simple weights, bias, and input for testing
    nn = _build_test_network()
    W_curr = np.array([[1.0, -1.0], [0.5, 2.0]])
    b_curr = np.array([[0.1], [-0.2]])
    A_prev = np.array([[1.0, 2.0], [-1.0, 0.0]])

    # perform a single forward pass with dummy values and define expected outputs manually
    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, "relu")
    expected_Z = np.array([[-0.9, 4.3], [-0.9, -0.7]])
    expected_A = np.array([[0.0, 4.3], [0.0, 0.0]])

    # assert that the computed values match the expected values
    assert np.allclose(Z_curr, expected_Z)
    assert np.allclose(A_curr, expected_A)


def test_forward():
    """
    Verify full forward pass output and cache values for a two-layer network.
    """
    # init dummy network with dummy weights and biases in the param dict
    nn = _build_test_network()
    nn._param_dict["W1"] = np.array([[1.0, 0.0], [0.0, 1.0]])
    nn._param_dict["b1"] = np.array([[0.0], [0.5]])
    nn._param_dict["W2"] = np.array([[1.5, -1.0]])
    nn._param_dict["b2"] = np.array([[0.1]])

    # define a simple inputs and perform a forward pass
    X = np.array([[1.0, 2.0], [0.5, -1.0]])
    y_hat, cache = nn.forward(X)

    # manually define expected values for each layer's output and activation
    z1_expected = np.array([[1.0, 2.5], [0.5, -0.5]])
    a1_expected = np.maximum(0.0, z1_expected)
    z2_expected = a1_expected @ nn._param_dict["W2"].T + nn._param_dict["b2"].T
    y_expected = 1.0 / (1.0 + np.exp(-z2_expected))

    # assert that the cache contains expected keys and the values match the expected values for each layer and final output
    assert set(cache.keys()) == {"A0", "A1", "A2", "Z1", "Z2"}
    assert np.allclose(cache["Z1"], z1_expected)
    assert np.allclose(cache["A1"], a1_expected)
    assert np.allclose(cache["Z2"], z2_expected)
    assert np.allclose(cache["A2"], y_expected)
    assert np.allclose(y_hat, y_expected)


def test_single_backprop():
    """
    Validate per-layer sigmoid backprop gradients against manual formulas.
    """
    # init dummy network along with dummy weights, biases, and other inputs for single backprop
    nn = _build_test_network()
    W_curr = np.array([[0.4, -0.6]])
    b_curr = np.array([[0.0]])
    Z_curr = np.array([[0.2], [-0.4]])
    A_prev = np.array([[1.0, 2.0], [0.5, -1.0]])
    dA_curr = np.array([[0.8], [-0.3]])

    # perform a single backprop with the dummy values and a sigmoid activation
    dA_prev, dW_curr, db_curr = nn._single_backprop(
        W_curr=W_curr,
        b_curr=b_curr,
        Z_curr=Z_curr,
        A_prev=A_prev,
        dA_curr=dA_curr,
        activation_curr="sigmoid",
    )

    # manually compute the expected 
    sig = 1.0 / (1.0 + np.exp(-Z_curr))
    dZ_expected = dA_curr * sig * (1.0 - sig)   # use sigmoid derivative
    dA_prev_expected = dZ_expected @ W_curr     # use backprop formula for dA_prev
    dW_expected = dZ_expected.T @ A_prev        # use backprop formula for dW
    db_expected = np.sum(dZ_expected, axis=0, keepdims=True).T  # use backprop formula for db

    # assert that the computed gradients match the expected values
    assert np.allclose(dA_prev, dA_prev_expected)
    assert np.allclose(dW_curr, dW_expected)
    assert np.allclose(db_curr, db_expected)


def test_predict():
    """
    Ensure predict returns the same output as the forward pass.
    """
    # init a dummy network and a simple dummy input X
    nn = _build_test_network()
    X = np.array([[0.2, 0.1], [0.4, -0.3], [0.0, 0.0]])

    # predict the output using the predict method
    y_pred = nn.predict(X)

    # predict the output using the forward method (more rigorously tested in test_forward;
    # both methods will fail if the forward pass method is incorrect)
    y_forward, _ = nn.forward(X)

    # assert that the predicted output matches the forward pass
    assert np.allclose(y_pred, y_forward)
    assert y_pred.shape == (3, 1)   # ensure output shape is correct for 3 samples


def test_binary_cross_entropy():
    """
    Check BCE loss value matches the analytical expression.
    """
    # init a dummy network (with BCE loss) and define dummy true labels and predictions
    nn = _build_test_network(loss_function="bce")
    y_true = np.array([[1.0], [0.0], [1.0]])
    y_hat = np.array([[0.9], [0.2], [0.7]])

    # compute the BCE loss with the method and manually
    loss = nn._binary_cross_entropy(y_true, y_hat)
    expected = -np.mean(y_true * np.log(y_hat) + (1.0 - y_true) * np.log(1.0 - y_hat))

    # assert that the computed loss matches the expected value
    assert np.allclose(loss, expected)


def test_binary_cross_entropy_backprop():
    """
    Check BCE gradient with respect to predictions is computed correctly.
    """
    # init a dummy network (with BCE loss) and define dummy true labels and predictions
    nn = _build_test_network(loss_function="bce")
    y_true = np.array([[1.0], [0.0], [1.0]])
    y_hat = np.array([[0.9], [0.2], [0.7]])

    # compute the BCE backprop gradients with the method and manually
    grad = nn._binary_cross_entropy_backprop(y_true, y_hat)
    expected = -(np.divide(y_true, y_hat) - np.divide(1.0 - y_true, 1.0 - y_hat)) / y_true.size

    # assert that the computed BCE backprop gradients match the expected values
    assert np.allclose(grad, expected)


def test_mean_squared_error():
    """
    Check MSE loss value matches the analytical expression.
    """
    # init a dummy network (with MSE loss) and define dummy true labels and predictions
    nn = _build_test_network(loss_function="mse")
    y_true = np.array([[1.0], [0.0], [1.0]])
    y_hat = np.array([[0.8], [0.3], [0.6]])

    # compute the MSE loss with the method and manually
    loss = nn._mean_squared_error(y_true, y_hat)
    expected = np.mean((y_true - y_hat) ** 2)

    # assert that the computed loss matches the expected value
    assert np.allclose(loss, expected)


def test_mean_squared_error_backprop():
    """
    Check MSE gradient with respect to predictions is computed correctly.
    """
    # init a dummy network (with MSE loss) and define dummy true labels and predictions
    nn = _build_test_network(loss_function="mse")
    y_true = np.array([[1.0], [0.0], [1.0]])
    y_hat = np.array([[0.8], [0.3], [0.6]])

    # compute the MSE backprop gradients with the method and manually
    grad = nn._mean_squared_error_backprop(y_true, y_hat)
    expected = 2.0 * (y_hat - y_true) / y_true.size

    # assert that the computed MSE backprop gradients match the expected values
    assert np.allclose(grad, expected)


def test_sample_seqs():
    """
    Verify oversampling balances positive and negative class counts.
    """
    # define simple seqs and dummy labels with class imbalance
    seqs = ["AAA", "TTT", "CCC", "GGG"]
    labels = [True, False, False, False]

    # sample the sequences
    np.random.seed(0)
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    # confirm that the class counts are balanced and length of seqs and labels match
    pos_count = sum(sampled_labels)
    neg_count = len(sampled_labels) - pos_count
    assert pos_count == neg_count
    assert len(sampled_seqs) == len(sampled_labels)

    # confirm that all sampled seqs are derived from the original set cooresponding labels are correct
    pos_original = {"AAA"}
    neg_original = {"TTT", "CCC", "GGG"}
    for seq, label in zip(sampled_seqs, sampled_labels):
        if label:
            assert seq in pos_original
        else:
            assert seq in neg_original


def test_one_hot_encode_seqs():
    """
    Check DNA one-hot encoding order and flattening for short sequences.
    """
    # encode simple sequences
    seqs = ["AT", "CG", "AGA"]
    encoded = one_hot_encode_seqs(seqs)

    # define expected encodings manually and test against output
    expected = [
        [1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    ]
    assert encoded == expected