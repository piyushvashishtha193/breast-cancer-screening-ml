"""Quick sanity checks: python -m pytest -q"""

from src.data import load_data, split
from src.models import candidates


def test_data_shape_and_target():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 212  # malignant cases


def test_split_is_stratified():
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = split(X, y)
    assert len(X_te) == 143
    assert abs(y_tr.mean() - y_te.mean()) < 0.01


def test_every_candidate_fits():
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = split(X, y)
    for name, (pipe, _) in candidates().items():
        pipe.fit(X_tr, y_tr)
        assert pipe.score(X_te, y_te) > 0.85, name


def test_data_audit_and_tests():
    from src.eda import audit, univariate_tests

    X, y = load_data()
    report = audit(X, y)
    assert report["missing_values"] == 0
    assert report["duplicate_rows"] == 0
    tests = univariate_tests(X, y)
    assert len(tests) == 30
    assert tests["p_bonferroni"].between(0, 1).all()


def test_threshold_meets_sensitivity_target():
    from src.train import pick_threshold

    X, y = load_data()
    X_tr, _, y_tr, _ = split(X, y)
    pipe, _ = candidates()["Logistic Regression"]
    threshold, sweep = pick_threshold(pipe, X_tr, y_tr, target=0.98)
    assert 0.05 <= threshold <= 0.95
    assert sweep.loc[sweep["threshold"] == threshold, "sensitivity"].iloc[0] >= 0.98
