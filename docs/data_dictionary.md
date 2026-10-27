# Data dictionary

Source: competition `metaData.csv`. Missing-value ratios are computed on train + test combined
(153,701 rows) and come from the competition notebook.

| Files | Rows | Columns |
|---|---|---|
| `train.csv` | 138,701 | 50 (includes `TargetValue`) |
| `test.csv` | 15,000 | 49 |
| `sample_submission.csv` | 15,000 | `TransactionID`, `TargetValue` |

## Identifiers, target and transaction

| Column | Description | Missing |
|---|---|---|
| `TransactionID` | Unique serial number of each accounting record | 0 % |
| `TargetValue` | **Target** — financial amount of the transaction (USD) | train only |
| `AssetID` | Internal tracking number of a specific piece of hardware | 0 % |
| `ProductConfigID` | System code for the technical build-out of the unit | 0 % |
| `DataOriginCode` | Software/platform the data originated from | 0 % |
| `VendorPartnerID` | External entity / third-party provider involved | 0 % |
| `TransactionDate` | Date the entry was finalised (1990-01-31 onwards) | 0 % |
| `RegionCode` | Geographic area / jurisdiction of the event | 0 % |

## Machine condition

| Column | Description | Missing |
|---|---|---|
| `ManufactureYear` | Year the asset was produced (`1000`/`1001` are placeholders) | 0 %* |
| `OperationalHoursMeter` | Lifetime run-time / work cycles (`0` usually means unrecorded) | 40.9 % |
| `UtilizationTier` | Low / Medium / High usage ranking | 63.7 % |

\* placeholder values are treated as missing during cleaning.

## Product specification

| Column | Description | Missing |
|---|---|---|
| `Spec_FullDescriptor` | Full alphanumeric model string (e.g. `950FII`, `PC200LC6`) | 0 % |
| `Spec_BaseClass` | Broad primary model category | 0 % |
| `Spec_SubClass` | Secondary, more specific tier | 26.7 % |
| `Spec_ReleaseSeries` | Production run / version | 76.3 % |
| `Spec_VariantModifier` | Revisions or custom tweaks to a standard model | 62.3 % |
| `AssetScaleFactor` | Physical size / capacity relative to a standard unit | 29.7 % |
| `FunctionalClassification` | Grouping by operational role | 0 % |
| `InventoryGroupCategory` | Highest-level inventory class | 0 % |
| `InventoryGroupDescription` | Text description of the inventory class | 0 % |

## Mechanical configuration

| Column | Description | Missing |
|---|---|---|
| `CabinType` | Operator station / enclosure design | 0 % |
| `DrivetrainType` | Power-transfer system to the movement components | 64.9 % |
| `Forks` | Lifting attachment module (legacy terminology) | 78.9 % |
| `col1` | Primary configuration of the drive system | 75.6 % |
| `col3` | Stabilisation subsystem | 79.0 % |
| `col4` | Layout / reach of the extension arm | 92.5 % |
| `col5` | Engine air-intake / aspiration | 92.5 % |
| `col6` | Status / type of the primary extension module | 83.1 % |
| `col7` | Horizontal width of the active working component | 83.1 % |
| `col8` | Protective housing / shielding | 83.1 % |
| `col9` | Rated power output | 83.1 % |
| `col10` | Hydraulic / fluid power system class | 7.5 % |
| `col11` | Load-bearing / shock-absorption interface | 83.1 % |
| `col12` | Piercing / penetrating accessories | 72.3 % |
| `col13` | Finishing / smoothing accessories | 83.1 % |
| `col14` | Actuator control logic and hardware | 83.1 % |
| `col15` | Wheel / tyre measurements | 69.5 % |
| `col16` | Coupling mechanism | 35.1 % |
| `col18` | Ground-traction system | 99.95 % — dropped |
| `col19` | Volumetric fluid flow rate | 99.95 % — dropped |
| `col20` | Tread / surface pattern of the traction system | 48.7 % |
| `col21` | Ground-contact width | 48.7 % |
| `col22` | Maximum reach of the operational arm | 48.7 % |
| `col23` | Material-retaining interface | 48.7 % |
| `col24` | Control-system configuration map | 48.7 % |
| `col25` | Profile of the traction components | 48.7 % |
| `col27` | Attached work-tool category | 89.3 % |
| `col28` | Directional steering input configuration | 89.3 % |
| `col29` | Differential type | 86.4 % |
| `col30` | Primary steering interface | 86.4 % |

Several groups of columns share an identical missing ratio (e.g. `col6`–`col14` at 83.1 %,
`col20`–`col25` at 48.7 %), which suggests each group is only recorded for one family of machines.
Missingness is therefore informative, which is why the pipeline keeps an explicit `Unknown` category
and `_miss` flags instead of discarding these columns.
