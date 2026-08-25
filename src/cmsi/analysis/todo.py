"""Analyses still to be written.

Each note says what the function should return. Delete the raise and write the
body when you get to it; nothing else imports these, so they can change shape
freely.

Implemented and moved out of here (2026-08-23, per the design audit):
    congruent/opposite classification, congruency balance, MSL lesion,
    RF shifts and gain fields          -> cmsi.analysis.units
    Kording-style behavioral curves    -> cmsi.analysis.behavior
"""


def indices(r_vis, r_prop, r_multi):
    """Per-unit additivity / response-enhancement / additivity-ratio indices.

    Compare each unit's multisensory response to its two unisensory ones.
    """
    raise NotImplementedError


def geometry(activations, post_c1):
    """Is p(C=1|x) an explicit axis of the population geometry?

    Participation ratio, representational dissimilarity, and how much variance
    lies along the p(C=1|x) direction.
    """
    raise NotImplementedError
