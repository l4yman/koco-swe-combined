def __getattr__(name):
    if name == "CodeIndex":
        from moatless.index.code_index import CodeIndex

        return CodeIndex
    if name == "IndexSettings":
        from moatless.index.settings import IndexSettings

        return IndexSettings
    raise AttributeError(name)


__all__ = ["CodeIndex", "IndexSettings"]
