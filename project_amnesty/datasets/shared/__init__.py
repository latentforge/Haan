"""Shared contract between the offline and runtime tiers.

The seam of the package: `schema.py` defines the unified Arrow schema that the
offline builders write (Sample.to_row) and the runtime loader reads
(row_to_arrays / _ListColumn). It belongs to neither tier -- both depend on it
directly, neither depends on the other.

Keep this minimal. Anything that only one tier needs belongs in that tier, not
here; a growing `shared/` is the sign the offline/runtime boundary is leaking.
"""
