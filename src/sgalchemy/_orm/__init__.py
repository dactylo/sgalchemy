"""Hold package-internal session modules and install the `Entity.find()` finder on import."""

from sgalchemy._orm._scoping import install_entity_finder_callback

install_entity_finder_callback()
