import numpy as np
import pandas as pd
import pytest
from spending.model import FEATURES, load_data, train, predict

def test_holdout_is_disjoint_and_reproducible():
    frame = load_data()
    model, report, training, holdout = train(frame)
    assert set(training.index).isdisjoint(holdout.index)
    assert len(training) + len(holdout) == len(frame)
    assert report == train(frame)[1]
    assert report["holdout"]["rmse"] < report["baseline"]["rmse"]
    assert np.isfinite(predict(model, frame[FEATURES])).all()

@pytest.mark.parametrize("problem", ["missing_column", "missing_value", "infinity", "duplicate", "negative"])
def test_invalid_data_rejected(tmp_path, problem):
    frame = load_data()
    if problem == "missing_column": frame = frame.drop(columns=FEATURES[0])
    if problem == "missing_value": frame.loc[0, FEATURES[0]] = np.nan
    if problem == "infinity": frame.loc[0, FEATURES[0]] = np.inf
    if problem == "duplicate": frame = pd.concat([frame, frame.iloc[[0]]])
    if problem == "negative": frame.loc[0, FEATURES[0]] = -1
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError): load_data(path)

def test_prediction_column_order_and_validation():
    frame = load_data()
    model = train(frame)[0]
    sample = frame[FEATURES].iloc[[0]]
    np.testing.assert_allclose(predict(model, sample), predict(model, sample[FEATURES[::-1]]))
    sample.iloc[0, 0] = np.inf
    with pytest.raises(ValueError): predict(model, sample)
