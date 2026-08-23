"""Analyses still to be written.

Each note says what the function should return. Delete the raise and write the
body when you get to it; nothing else imports these, so they can change shape
freely.
"""


def indices(r_vis, r_prop, r_multi):
    """Per-unit additivity / response-enhancement / additivity-ratio indices.

    Compare each unit's multisensory response to its two unisensory ones.
    """
    raise NotImplementedError


def geometry(activations, p_common):
    """Is p(C=1) an explicit axis of the population geometry?

    Participation ratio, representational dissimilarity, and how much variance
    lies along the p(C=1) direction.
    """
    raise NotImplementedError


def rf_shifts(activations, eye, x_vis_body):
    """Do preferred locations shift with eye position?

    Shift gain ~ 1 means a retinal code, ~ 0 a body-frame code; partial shifts
    are the interesting case.
    """
    raise NotImplementedError


def congruent_opposite(activations, x_vis_body, x_prop):
    """Classify units as congruent vs opposite, then ask whether the balance
    between the two populations predicts the readout's fusion weight."""
    raise NotImplementedError


def ablation(model, X, unit_mask):
    """Silence sub-populations of MSL and look for a dissociation: does the
    location estimate degrade while the implied p(C=1) survives, or vice versa?"""
    raise NotImplementedError



#another to do coming from my discussion with claude :
# Keep the SIL/MSL structure and unit-level analyses intact, and add two: classify MSL units by congruency
# (correlate each unit's response to visual-position sweeps vs. proprioceptive-position sweeps, Rideaux's Eq. 11 logic)
# to test whether congruent/opposite subpopulations emerge, and test whether the congruent–opposite activity balance predicts the network('s p(C) output.
# If it does, 'you')d be showing that the mechanism Rideaux found
# in a visual-vestibular fusion/scission network also emerges when a network is trained directly on the Bayes-optimal mixture and
# must simultaneously solve a coordinate transformation — that's the distinctive contribution of this design.