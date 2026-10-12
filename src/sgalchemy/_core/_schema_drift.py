"""Internal schema-generation drift signal shared by core planning surfaces."""


class SchemaDrift(BaseException):
    """Report one schema fact that a bounded caller may refresh and retry.

    Instances retain only actionable context, not a registry binding.

    Attributes:
        entity_type: Entity type or operation owning the stale fact.
        field_code: Field code or wildcard locating the stale fact.
        target_type: Schema requirement the captured snapshot does not satisfy.
    """

    def __init__(self, entity_type: str, field_code: str, target_type: str) -> None:
        """Initialize actionable context without retaining a binding object.

        Args:
            entity_type: Entity type or operation owning the stale fact.
            field_code: Field code or wildcard locating the stale fact.
            target_type: Schema requirement the captured snapshot does not
                satisfy.
        """
        super().__init__(f'{entity_type}.{field_code} declares unavailable target {target_type}')
        self.entity_type = entity_type
        self.field_code = field_code
        self.target_type = target_type
