# puretd

This folder is taken from [PureTD](https://github.com/alexstrehl/puretd)
(MIT licence, see `LICENSE`) at commit `b2750df88b62400b35ed1bf73261748eec33a7de`. It holds the
seven modules that play the PureTD `cubeless_prob5_512_512_256_128.pt` network, which the
oracle-strength round-robin uses as the agent `puretd` (checkpoint `models/puretd.pt`).

Edits to the upstream files:

- `__init__.py` is new, so the folder imports as a package.
- Imports between the seven modules are relative (`from .model import ...` instead of
  `from model import ...`). No other line is changed.
