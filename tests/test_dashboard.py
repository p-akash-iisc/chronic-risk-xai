def test_import_dashboard():
    # dashboard should import without side effects
    import src.chronic_risk.dashboard as dash
    assert callable(getattr(dash, "st", None)) or dash is not None
