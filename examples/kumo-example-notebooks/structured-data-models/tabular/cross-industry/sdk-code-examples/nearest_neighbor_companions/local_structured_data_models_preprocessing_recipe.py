"""Shared stable-category recipe for nearest-neighbor examples."""

import sdm
import sdm.processing as sp


def knn_recipe() -> sdm.Recipe:
    """Keep target class identities stable across retrieved contexts."""
    recipe = sdm.models.TabICLv2.default_recipe()
    recipe.target = sp.StypeDispatch(categorical=sp.AlignCategories(sort_by="value"))
    return recipe
