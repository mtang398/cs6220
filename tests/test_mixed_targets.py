"""Independent checks of the controlled spectral-target workload."""
import json
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist

from nystrom_project.datasets import mixed_eigenspace_targets, raw_data, prepare


def test_mixed_targets_preserve_inputs_and_spectral_components(tmp_path):
    X, smooth = raw_data(tmp_path, 'synthetic', 0, 'smooth', 80)
    Xm, y = raw_data(tmp_path, 'synthetic', 0, 'mixed', 80)
    assert np.array_equal(X, Xm)
    assert not np.allclose(y, smooth)
    order = np.random.default_rng(np.random.SeedSequence([0, 6220, 12])).permutation(80)
    train, test = order[:64], order[64:]
    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    Z, Ztest = (X[train]-mean)/std, (X[test]-mean)/std
    h = np.median(pdist(Z))
    K = np.exp(-cdist(Z, Z, 'sqeuclidean')/(2*h*h))
    values, V = eigh(K)
    values, V = values[::-1], V[:, ::-1]
    components = V.T@y[train]
    selected = np.r_[np.arange(1, 9), np.arange(16, 24)]
    outside = np.setdiff1d(np.arange(64), selected)
    assert np.linalg.norm(components[outside]) < 1e-10
    np.testing.assert_allclose(np.sum(components[1:9]**2), 32., rtol=1e-12)
    np.testing.assert_allclose(np.sum(components[16:24]**2), 32., rtol=1e-12)
    assert abs(y[train].mean()) < 1e-12
    # Reconstruct held-out values from independently recovered spectral coefficients.
    weights = V[:, selected]@(components[selected]/values[selected])
    expected = np.exp(-cdist(Ztest, Z, 'sqeuclidean')/(2*h*h))@weights
    np.testing.assert_allclose(y[test], expected, rtol=1e-10, atol=1e-10)
    changed = X.copy(); changed[test] += 5.
    altered = mixed_eigenspace_targets(changed, 0)
    np.testing.assert_array_equal(altered[train], y[train])


def test_mixed_reference_is_regularization_independent_and_has_provenance(tmp_path):
    (tmp_path/'data').mkdir()
    first, config = prepare(tmp_path, 'synthetic', 0, .001, 'mixed', 80)
    second, _ = prepare(tmp_path, 'synthetic', 0, .1, 'mixed', 80)
    with np.load(first) as a, np.load(second) as b:
        for key in ('X_train', 'X_test', 'y_train', 'y_test', 'bandwidth'):
            np.testing.assert_array_equal(a[key], b[key])
        assert not np.allclose(a['alpha_reference'], b['alpha_reference'])
        saved = json.loads(str(a['config']))
        assert saved['target_construction']['eigenvectors_available_to_solver'] is False
        assert saved['reference_relative_residual'] < 1e-10
        assert not any('basis' in key or 'eigenvector' in key for key in a.files)
    assert config['target_construction']['descending_zero_based_bands'] == [[1, 9], [16, 24]]
