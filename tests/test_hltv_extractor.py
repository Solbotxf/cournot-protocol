from agents.collector.extractors import find_extractor


def test_hltv_extractor_registry_match():
    url = "https://www.hltv.org/stats/players/18987/b1t?event=cluj-napoca-2026"
    ext = find_extractor(url)
    assert ext is not None
    assert ext.source_id == "hltv"
