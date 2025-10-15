import pandas as pd
from src.chronic_risk.features import add_domain_features

def test_add_domain_features():
    df = pd.DataFrame({"a":[1,2,3]})
    out = add_domain_features(df)
    assert len(out) == 3
