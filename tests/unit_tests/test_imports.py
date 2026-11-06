from langchain_optiq import __all__

EXPECTED = ["ChatOptiQ", "__version__"]


def test_all_imports() -> None:
    assert sorted(EXPECTED) == sorted(__all__)
