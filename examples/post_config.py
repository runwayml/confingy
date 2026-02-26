from confingy import Lazy, track


@track
class ChildModule:
    def __init__(self, hidden_size: int, name: str | None = None):
        self.hidden_size = hidden_size
        self.name = name


@track
class ParentModel:
    def __init__(
        self,
        hidden_size: int,
        name: str,
        encoder: Lazy[ChildModule],
        decoder: Lazy[ChildModule],
    ):
        self.hidden_size = hidden_size
        self.name = name
        self.encoder = encoder.instantiate()
        self.decoder = decoder.instantiate()

    @classmethod
    def __post_config__(cls, instance: Lazy, changed_key: str | None) -> Lazy:
        # Propagate values to child modules only when relevant params change
        # changed_key is None on initial creation, otherwise it's the param that changed
        if changed_key is None or changed_key == "name":
            instance.encoder.name = f"{instance.name}_encoder"
            instance.decoder.name = f"{instance.name}_decoder"
        if changed_key is None or changed_key == "hidden_size":
            instance.encoder.hidden_size = instance.hidden_size
            instance.decoder.hidden_size = instance.hidden_size
        return instance


lazy_parent = ParentModel.lazy(
    hidden_size=1_024,
    name="my_model",
    encoder=ChildModule.lazy(hidden_size=0),
    decoder=ChildModule.lazy(hidden_size=0),
)
print(lazy_parent.encoder.hidden_size)  # 1024
print(lazy_parent.encoder.name)  # "my_model_encoder"
print(lazy_parent.decoder.hidden_size)  # 1024
print(lazy_parent.decoder.name)  # "my_model_decoder"

lazy_parent.hidden_size = 512
print(lazy_parent.encoder.hidden_size)  # 512
print(lazy_parent.decoder.hidden_size)  # 512

lazy_parent.name = "new_model"
print(lazy_parent.encoder.name)  # "new_model_encoder"
print(lazy_parent.decoder.name)  # "new_model_decoder"
