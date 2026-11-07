from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_prediction_and_extrapolation():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py")).run(timeout=30)
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert any(m.label == "Estimated yearly spending" for m in app.metric)
    app.number_input[0].set_value(1000.0)
    app.button[0].click().run()
    assert not app.exception
    assert any("Outside the training range" in w.value for w in app.warning)
