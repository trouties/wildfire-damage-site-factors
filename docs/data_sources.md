# Data sources

None of the sources needs credentials. Every download step writes `manifest.json` next to its files (URL, SHA-256, retrieval time, licence); the Sentinel-2 step, which stores no files, writes the item and asset URLs instead. The retrieval dates below are taken from those manifests. Endpoints and licence strings are in `config/data_sources.yaml`.

| Data | Product and how it is selected | Script | Retrieved | Licence / attribution |
|---|---|---|---|---|
| Damage grades, AOIs, burnt area | Copernicus EMS Rapid Mapping grading (GRA) vector packages of every Greek wildfire activation since 2019-01-01. Layers used: `builtUpP` (graded building points), `areaOfInterestA`, `observedEventA` (burnt area). | `src.acquire.ems` | 2026-11-13 | Copernicus data policy, Regulation (EU) 2021/696. © European Union, Copernicus Emergency Management Service |
| Building footprints | Microsoft Global ML Building Footprints, Greece tiles (upload date 2026-02-23), clipped to each exposure domain's bounding box plus 300 m | `src.acquire.buildings` | 2026-11-13/14 | ODbL 1.0 |
| Elevation | Copernicus GLO-30 DEM tiles from the AWS Open Data bucket | `src.acquire.dem` | 2026-04-06 (3 tiles), 2026-11-14 (the rest) | © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA |
| Land cover | CORINE Land Cover 2018 vector, EEA map service, each exposure domain's bounding box plus 500 m | `src.acquire.corine` | 2026-11-14 | © European Union, Copernicus Land Monitoring Service 2018, European Environment Agency (EEA) |
| Land cover at 10 m (one check) | ESA WorldCover 2020 v100 map tiles (3 x 3 degrees) from the AWS Open Data bucket, every tile under an exposure domain's bounding box plus 510 m | `src.acquire.worldcover` | 2026-11-14 | CC BY 4.0. © ESA WorldCover project 2020 / Contains modified Copernicus Sentinel data (2020) processed by ESA WorldCover consortium |
| Burn severity (control) | Sentinel-2 L2A Collection 1 (`sentinel-2-c1-l2a`) from the Earth Search STAC catalogue; bands B8A, B12 and SCL read as windows from the Cloud-Optimized GeoTIFFs on AWS, nothing stored. One pre- and one post-fire acquisition per event by the rules in `config/features.yaml`; the chosen items are in `outputs/sentinel2/scenes.csv`, their URLs in the manifest | `src.features.dnbr` | 2026-11-14 | Copernicus Sentinel data terms; contains modified Copernicus Sentinel data |

## From packages to events

`src.acquire.ems` lists the activations through the EMS activations API. From EMSR656 (2023) on, the rapid-mapping API lists each activation's products; for older activations the package links are read from the activation's portal page. Package names are parsed by `PKG_RE` in that module.

An event is one graded AOI. Monitoring products and re-releases of the same AOI are merged point by point, newest first (`version_merge_tolerance_m` in `config/events.yaml`). The selection rules in `config/events.yaml` use label availability only; `outputs/labels/ems_events.csv` gives the reason for every AOI and `outputs/labels/label_summary.csv` the counts and link rates of the selected ones.

Not used: EMS delineation (DEL) products, which have no building grades; EMS activations before 2019, which predate CORINE 2018.

## What an EMS building grade means

- **Classes.** The EMS Rapid Mapping manual ([damage assessment](https://mapping.emergency.copernicus.eu/about/rapid-mapping-manual/detection-methods-damage-assessment/), page dated 2025-11-12) equates *Destroyed* with EMS-98 grades 4-5 and *Damaged* with grades 1-3. *Possibly damaged* "refers to cases when the confidence level of the interpretation is slightly lower (e.g. bad image quality)". The grades are to be "intended as a proxy and near-real time estimation for damage, and not as ground truth data". The map in the EMSR909 package (`Maps/EMSR909_AOI01_GRA_PRODUCT_62000_map_v1.pdf`) footnotes *Possibly damaged* as "Presence of damage proxies and proximity with destroyed/damaged asset".
- **Which buildings are graded.** Neither the manual nor the package metadata says whether `builtUpP` holds every building in the AOI or the burnt area. Two study products also grade buildings away from the fire: EMSR598 grades 30,255 buildings *No visible damage* and EMSR607 358 (`ems_events.csv`), yet only one footprint in each exposure domain carries that grade (`label_summary.csv`): in these products *No visible damage* marks buildings away from the burnt area, and inside it EMS graded buildings *Possibly damaged* or worse. The other twelve products list *Possibly damaged*, *Damaged* and *Destroyed* only. Footprints that no EMS point falls on include large buildings as well as small structures; the README results block gives the share.
- **Where the points come from.** The `builtUpP` metadata describes the layer as "approximate building footprints as digitized from imagery and/or extracted from external sources (as indicated in the OR_SRC_ID field)". Each package's `source` table names the source behind every `or_src_id`. In the study events these are OpenStreetMap, digitisation on a pre- or post-event image, and, for EMSR909, Overture Maps. `src.acquire.ems` attaches the source name to every point, and `label_summary.csv` gives, per event, the share of damaged (`osm_share_damaged`) and of possibly damaged points (`osm_share_possibly`) in the exposure domain whose geometry EMS took from OpenStreetMap.

## Sentinel-2 collection

Collection 1 is ESA's reprocessing of the archive with one processing baseline; every item used declares the same reflectance scale and offset, which `src.features.dnbr` checks. Older L2A products before baseline 04.00 (January 2022) have no offset, so a pre/post pair taken from the original archive across that date would shift dNBR. On 2026-11-14 the Earth Search Collection 1 held no acquisition in the post-fire windows of the two 2022 study events (Mount Penteli, Vatera); `scenes.csv` records them as missing. The original-archive collection on the same catalogue was not used to fill the gap.

## Committed derived data

None of the downloads is committed. `data/analysis_table.csv`, written by `src.features.build`, has one row per building in a selected event's exposure domain, with its EMS grade and label, footprint centroid (EPSG:2100) and area, the static features and the burn-severity control. It is derived from all sources above and released under ODbL 1.0, the licence of the footprints. `src.analysis.evaluate` reads only this table and `outputs/labels/label_summary.csv`.
