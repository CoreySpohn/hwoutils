# Internal reference guard

`hwoutils.internal_refs` owns the shared repository text guard. It uses the
Python standard library; the optional pytest integration requires pytest.
Install hwoutils in the environment that runs pytest and pre-commit.

Add these three lines beside an existing `pytest_plugins` list in `conftest.py`:

```python notest
import importlib.util

if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")
```

Create `pytest_plugins = []` first if the file does not already define it.
Remove the repository's copied guard script and guard test. The plugin adds
one test named `internal-refs`, including when pytest selects individual files.
It scans existing tracked text files under the pytest root directory, ignores
unstaged deletions, and skips `CHANGELOG.md`. Supported suffixes are `.py`,
`.md`, `.rst`, `.toml`, `.yaml`, `.yml`, `.cff`, `.ipynb`, and `.txt`.
With xdist's default scheduling, the item executes on one worker.

Use `--deselect=internal-refs` to deselect the item, or
`-p no:hwoutils.internal_refs` to disable the plugin.

Add this local hook to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: no-internal-refs
        name: no internal research references
        entry: python -m hwoutils.internal_refs
        language: system
        types: [text]
        exclude: ^(CHANGELOG\.md|tools/check_internal_refs\.py)$
```

The exclusion preserves the existing hook configuration while repositories
remove the copied script. Explicit filenames are scanned regardless of suffix;
without filenames, `python -m hwoutils.internal_refs` scans tracked text files
in the current repository. Clean scans exit with status 0; hits exit with
status 1 and report the filename, line number, pattern and offending line.
Append `internal-ref-ok` to a deliberately functional line to waive that line.
