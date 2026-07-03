# GroundTruth

**A privacy-preserving LiDAR sensing-to-MRV system that verifies climate-smart farm practices from worker motion, without a camera.**

GroundTruth reads the *motion of farm labour* with a single low-cost, fixed LiDAR sensor and turns it into a verified record of which climate-smart practice was performed. It records geometry and movement as 3D point clouds, never faces or images, so it protects privacy by design. The goal is to make the climate work of smallholder farmers, much of it done by women, measurable and creditable for the first time, as ground evidence for MRV (Monitoring, Reporting and Verification).

This repository is open by default. Its first dataset is **HOMI**, a ground-level LiDAR dataset of agricultural worker activity.

---

## Why this exists

Climate finance follows the land title and the satellite image. Satellites see canopy from orbit but cannot tell rain from human effort, and payments reward whoever owns the land. The people doing the physical climate work stay uncounted. GroundTruth measures the labour itself, so the work becomes the unit that gets counted.

## How it works (four steps)

1. **Sense** — a single fixed LiDAR sensor at a plot edge captures farming motion as 3D point clouds. No camera.
2. **Recognise** — a lightweight machine-learning model classifies each practice (e.g. a hand-tool strike) and separates it from ordinary movement.
3. **Record** — the result becomes a simple verified record of which practice was done, owned by the farming cooperative.
4. **Verify (MRV)** — that record serves as the ground evidence for carbon and results-based climate finance.

## Repository structure

```
groundtruth/
├── README.md            ← you are here (the system)
├── LICENSE              ← MIT (code)
├── docs/                ← project documentation
│   └── DPG_CHECKLIST.md
├── datasets/
│   └── HOMI/            ← first dataset (see its own README)
├── models/              ← trained classification models
└── src/                 ← feature-extraction and classification code
```

## Datasets

| Dataset | Description | Location |
|---|---|---|
| **HOMI** | First release. Ground-level LiDAR dataset of agricultural worker activity. 16,676 point-cloud frames, 6 activity classes, 95.1% 5-fold CV accuracy. | [`datasets/HOMI/`](datasets/HOMI/) |

Future field datasets (e.g. from Sahel cooperatives) will be added under `datasets/` and remain owned by the collecting community, shared only as non-identifying geometry, with consent.

## Licensing

- **Code** in this repository: [MIT](LICENSE).
- **Datasets** (including HOMI): [Creative Commons Attribution 4.0 (CC BY 4.0)](datasets/HOMI/LICENSE-DATA). Use freely with attribution.

## How to cite

> Seo, H. (2026). *GroundTruth: privacy-preserving LiDAR sensing-to-MRV for farm labour. HOMI dataset (first release).* https://github.com/HJ-SEO-UL/groundtruth-dataset

## Contact

Hanju Seo — EarthCode — https://hanjuseo.com/HOMI

---

## 한글 요약

GroundTruth는 카메라가 아닌 저가 고정형 라이다로 농사 노동의 움직임을 읽어, 어떤 기후 농법이 실제로 이루어졌는지를 검증된 기록으로 바꾸는 시스템입니다. 얼굴이나 영상이 아니라 형태와 움직임만 담아 프라이버시를 지킵니다. 위성이 위에서 결과만 볼 때, GroundTruth는 지상에서 노동 자체를 재어, 그동안 세어지지 않던 여성 소농의 기후 노동을 측정 가능하고 재정 지원 가능하게 만듭니다. 이 저장소는 기본이 공개이며, 첫 번째 데이터셋은 지상 라이다 농업 활동 데이터셋 **HOMI**입니다.
