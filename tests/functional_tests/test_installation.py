import importlib.util
def test_installation():
    import HousePricePrediction
    import HousePricePrediction.ingest_data
    import HousePricePrediction.score
    import HousePricePrediction.train
    package_name = 'HousePricePrediction'
    assert importlib.util.find_spec(package_name) is not None
