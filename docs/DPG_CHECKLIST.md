# Digital Public Goods (DPG) self-check

The AICA Award scores how well an open-source solution aligns with the Digital
Public Goods Standard. This is a working self-check for GroundTruth against the
nine DPG indicators.

| # | DPG indicator | GroundTruth status | Evidence |
|---|---|---|---|
| 1 | Relevance to an SDG | Yes | SDG 13 (climate), SDG 5 (gender), SDG 2 (food) |
| 2 | Open licensing | Yes | Code MIT, data CC BY 4.0 |
| 3 | Clear ownership | Yes | Hanju Seo / EarthCode, stated in README and LICENSE |
| 4 | Platform independence | Yes | Runs offline, no proprietary cloud dependency |
| 5 | Documentation | Yes | README, dataset card, metadata, results |
| 6 | Non-PII data / data privacy | Yes | Geometry only, no faces or RGB; privacy checklist in metadata.md |
| 7 | Adherence to standards / best practices | Partial | Add: data format spec; test/benchmark script in src/ |
| 8 | Do no harm by design | Yes | Community data ownership, consent, no surveillance imagery |
| 9 | Do no harm (privacy, laws, content) | Yes | Coarse location only; no personal data |

## To close before submission
- [ ] Publish the repository as **Public**.
- [ ] Confirm both licenses are present (LICENSE and datasets/HOMI/LICENSE-DATA).
- [ ] Add the feature-extraction and training code to src/.
- [ ] Complete the privacy checklist in datasets/HOMI/metadata.md.
- [ ] Replace `<owner>` in the citation lines with the real GitHub owner.
