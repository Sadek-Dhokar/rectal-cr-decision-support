from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_app_loads_complete_example():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception
    markdown = "\n".join(str(item.value) for item in app.markdown)
    assert "COMPLETE RESPONSE" in markdown


def test_app_mri_unavailable_is_uncertain():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    app.selectbox[0].select("5. PT_0308 - COMPLETE RESPONSE")
    app.checkbox[3].set_value(False).run()
    assert not app.exception
    markdown = "\n".join(str(item.value) for item in app.markdown)
    assert "UNCERTAIN" in markdown


def test_app_ct_progression_shows_rule():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    app.radio[0].set_value("Manual entry").run()
    ct_field = next(
        field for field in app.selectbox
        if getattr(field, "label", "") == "ct_distant_progression"
    )
    ct_field.set_value("1.0").run()
    app.run()
    assert not app.exception
    markdown = "\n".join(str(item.value) for item in app.markdown)
    assert "NOT COMPLETE" in markdown
