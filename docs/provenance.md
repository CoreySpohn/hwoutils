# Build stamps and model reads

A build stamp records a fact about a product at the time it was built. The
`hwoutils.provenance` helpers encode facts as typed text and collect them as
sorted `(label, value)` pairs. On an Equinox module, store these pairs as static
metadata. A label can be dotted to keep each child's facts under its own prefix.

Compare stamps when assembling products that share a build requirement. A fact
is matched when both builds record the same known value, and mismatched when
they record different known values. A missing label or a value recorded as
`None` is unverified. The caller decides whether it has enough information to
proceed. Use `require_stamps` to raise on mismatches and get the unverified labels
back. The short `display_key` is suitable for messages.

A product can also record the model nodes it read during construction. Qualify each
path with the owner's name. To check whether a write makes the product stale, bind
those reads to their paths on the model. An owner held outside that model has no
writable path, so use `None` for its root.

```python
from hwoutils import provenance

built = provenance.build_stamps(band_nm=550, source=None)
requested = provenance.build_stamps(band_nm=600, source="calibration")
comparison = provenance.compare_stamps(built, requested)
assert comparison.mismatched == (("band_nm", "int:550", "int:600"),)
assert comparison.unverified == ("source",)

reads = provenance.qualify("scene", ("star.spectrum",))
reads += provenance.qualify("sampling", ("grid",))
bound = provenance.bind_reads(reads, {"scene": "", "sampling": None})
assert bound == ("star.spectrum",)
assert provenance.stale_reads(bound, ("star",)) == (("star", "star.spectrum"),)
```
