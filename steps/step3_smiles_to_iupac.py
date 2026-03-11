#!/usr/bin/env python3
"""
STEP 3 — SMILES → IUPAC Name Resolution  (V2 — Full Rebuild)
=============================================================

Resolves IUPAC names for every molecule in ToxCast, Tox21 and ClinTox.
Uses RDKit for SMILES canonicalisation and stereochemistry detection,
then resolves names through a three-API cascade:
    PubChem PUG REST → ChemSpider v2 → NCI CIR

Does NOT touch common_molecules_final.csv or t3db_processed.csv.

PIPELINE
--------
Phase 1 — SMILES Preprocessing
    • Parse raw SMILES with RDKit
    • Strip salts / multi-fragment notation (keep largest fragment)
    • Canonicalise with isomericSmiles=True  (PRESERVES stereochemistry)
    • Collect unique canonical SMILES to minimise API calls

Phase 2 — API Resolution  (one lookup per unique canonical SMILES)
    1. Local cache check   (data/step3_cache.csv)
    2. PubChem PUG REST    (POST, accepts any SMILES length)
    3. ChemSpider v2       (needs API key)
    4. NCI CIR             (GET fallback)
    5. Failed → mark as unresolved

Phase 3 — Collision Detection & Resolution
    Group resolved names; if >1 canonical SMILES → same name = collision.
    • Same flat SMILES → stereoisomers → add (R)/(S), (E)/(Z) prefix
    • Different flat SMILES → constitutional isomer with same name → flag
    • Unresolvable → data/failed_resolve.csv, iupac_name left blank

Phase 4 — Apply to Datasets
    Map raw_smiles → canonical_smiles → final_iupac_name
    Write [smiles, iupac_name, is_toxic] to each _final.csv
    Collisions/failures get iupac_name = "" (blank)

Stereochemistry Resolution:
    Type             | What Changes            | Naming Prefix
    Constitutional   | Connectivity            | Normal IUPAC
    Geometric        | Double bond orientation | (E)/(Z)
    Optical          | Chiral centre           | (R)/(S)
    Multiple centres | Several stereocentres   | (2R,3S,...)
    Conformers       | Rotation                | Not separately named

Inputs:
    data/toxcast_final.csv     (smiles [, iupac_name], is_toxic)
    data/tox21_final.csv       (smiles [, iupac_name], is_toxic)
    data/clintox_final.csv     (smiles [, iupac_name], is_toxic)
    data/herg_final.csv        (smiles [, iupac_name], is_toxic)
    data/dili_final.csv        (smiles [, iupac_name], is_toxic)

Outputs (overwritten in-place):
    data/toxcast_final.csv     (smiles, iupac_name, is_toxic)
    data/tox21_final.csv       (smiles, iupac_name, is_toxic)
    data/clintox_final.csv     (smiles, iupac_name, is_toxic)
    data/herg_final.csv        (smiles, iupac_name, is_toxic)
    data/dili_final.csv        (smiles, iupac_name, is_toxic)
    data/step3_cache.csv       (canonical_smiles, iupac_name, api_source)
    data/failed_resolve.csv    (canonical_smiles, flat_smiles, api_name, reason)

Run from project root:
    python steps/step3_smiles_to_iupac.py                  # full pipeline
    python steps/step3_smiles_to_iupac.py --dry-run         # Phase 1 stats only
    python steps/step3_smiles_to_iupac.py --cache-only      # Phase 1 + apply cache
    python steps/step3_smiles_to_iupac.py --limit 200       # resolve first 200 uncached
    python steps/step3_smiles_to_iupac.py --dataset toxcast # one dataset only
"""

import os
import re
import sys
import time
import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from urllib.parse import quote

import pandas as pd
import requests

# ── RDKit (SMILES canonicalisation + stereo detection) ────────────────
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)   # suppress warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════

DATA_DIR   = "./data"
CACHE_FILE = os.path.join(DATA_DIR, "step3_cache.csv")
FAILED_FILE = os.path.join(DATA_DIR, "failed_resolve.csv")

DATASETS = {
    "toxcast": os.path.join(DATA_DIR, "toxcast_final.csv"),
    "tox21":   os.path.join(DATA_DIR, "tox21_final.csv"),
    "clintox": os.path.join(DATA_DIR, "clintox_final.csv"),
    "herg":    os.path.join(DATA_DIR, "herg_final.csv"),
    "dili":    os.path.join(DATA_DIR, "dili_final.csv"),
}

# ── API endpoints ────────────────────────────────────────────────────
PUBCHEM_POST_URL      = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/IUPACName/JSON"
CHEMSPIDER_FILTER_URL = "https://api.rsc.org/compounds/v1/filter/smiles"
CHEMSPIDER_RESULT_URL = "https://api.rsc.org/compounds/v1/filter/{}/results"
CHEMSPIDER_RECORD_URL = "https://api.rsc.org/compounds/v1/records/{}/details?fields=IupacName"
NCI_CIR_URL           = "https://cactus.nci.nih.gov/chemical/structure/{}/iupac_name"

CHEMSPIDER_API_KEY = os.environ.get(
    "CHEMSPIDER_API_KEY",
    "$$",
)

# Rate limiting (PubChem: ≤5 req/s, ≤400 req/min)
PUBCHEM_DELAY   = 0.22   # seconds between PubChem calls
CHEMSPIDER_DELAY = 0.50
NCI_DELAY        = 0.35
SAVE_INTERVAL    = 100   # save cache every N resolutions


# ═════════════════════════════════════════════════════════════════════
# Phase 1 — SMILES Preprocessing  (RDKit)
# ═════════════════════════════════════════════════════════════════════

def preprocess_smiles(raw_smiles: str):
    """Parse, salt-strip, and canonicalise a raw SMILES.

    Returns (canonical_iso, canonical_flat, mol) or None on failure.
        canonical_iso:  stereo-preserving canonical SMILES
        canonical_flat: stereo-stripped canonical SMILES (for collision grouping)
        mol:            RDKit Mol object
    """
    if not raw_smiles or not isinstance(raw_smiles, str):
        return None

    raw_smiles = raw_smiles.strip()
    if not raw_smiles:
        return None

    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return None

    # ── Strip salts / multi-fragment notation ─────────────────────
    # Keep the fragment with the most heavy atoms.
    # "N=C(N)NN.N=C(N)NN"  →  single "N=C(N)NN"
    # "CC(=O)[O-].[Na+]"   →  "CC(=O)[O-]"
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if frags and len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

    # ── Assign stereochemistry so stereo queries work ─────────────
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # ── Canonicalise ──────────────────────────────────────────────
    canonical_iso  = Chem.MolToSmiles(mol, isomericSmiles=True)
    canonical_flat = Chem.MolToSmiles(mol, isomericSmiles=False)

    return canonical_iso, canonical_flat, mol


def get_stereo_info(mol):
    """Extract stereochemistry from an RDKit mol.

    Returns dict:
        chirals:  [(atom_idx, 'R'/'S'), ...]
        ez:       [(bond_idx, 'E'/'Z'), ...]
        has_stereo: bool
    """
    chirals = Chem.FindMolChiralCenters(mol, includeUnassigned=False,
                                         useLegacyImplementation=False)

    ez = []
    for bond in mol.GetBonds():
        st = bond.GetStereo()
        if st == Chem.rdchem.BondStereo.STEREOE:
            ez.append((bond.GetIdx(), "E"))
        elif st == Chem.rdchem.BondStereo.STEREOZ:
            ez.append((bond.GetIdx(), "Z"))

    return {
        "chirals": chirals,
        "ez": ez,
        "has_stereo": bool(chirals or ez),
    }


def build_stereo_prefix(mol):
    """Build a human-readable stereo prefix from RDKit analysis.

    Examples:  "(R)-",  "(2S,3R)-",  "(E)-",  "(R,E)-"

    Uses the atom-index-based chiral centres; index+1 gives a numeric
    locant (not identical to the IUPAC locant but sufficient to
    disambiguate stereoisomers of the same base name).
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    chirals = Chem.FindMolChiralCenters(mol, includeUnassigned=False,
                                         useLegacyImplementation=False)

    ez = []
    for bond in mol.GetBonds():
        st = bond.GetStereo()
        if st == Chem.rdchem.BondStereo.STEREOE:
            ez.append("E")
        elif st == Chem.rdchem.BondStereo.STEREOZ:
            ez.append("Z")

    parts = []
    # E/Z first (geometric isomerism)
    for label in ez:
        parts.append(label)
    # Then R/S with locant numbers
    for atom_idx, tag in sorted(chirals):
        locant = atom_idx + 1          # approximate IUPAC locant
        parts.append(f"{locant}{tag}")

    if not parts:
        return ""
    return "(" + ",".join(parts) + ")-"


def collect_and_preprocess(datasets_to_process: dict):
    """Load all datasets, preprocess every SMILES.

    Returns:
        raw_to_canon:  dict[raw_smiles → canonical_iso]
        canon_to_mol:  dict[canonical_iso → RDKit Mol]
        canon_to_flat: dict[canonical_iso → canonical_flat]
        parse_failed:  list[raw_smiles]    (RDKit couldn't parse)
        per_dataset:   dict[ds_name → DataFrame]  (original data)
    """
    raw_to_canon  = {}
    canon_to_mol  = {}
    canon_to_flat = {}
    parse_failed  = []
    per_dataset   = {}

    for ds_name, path in datasets_to_process.items():
        if not os.path.exists(path):
            logger.warning(f"  {path} not found — skip (run step 2 first)")
            continue
        df = pd.read_csv(path)
        # Keep only smiles + is_toxic (ignore old iupac_name if present)
        if "smiles" not in df.columns or "is_toxic" not in df.columns:
            logger.error(f"  {path}: missing 'smiles' or 'is_toxic' column")
            continue
        df = df[["smiles", "is_toxic"]].copy()
        df["smiles"] = df["smiles"].astype(str).str.strip()
        per_dataset[ds_name] = df

        # Preprocess each unique SMILES
        unique_smiles = df["smiles"].dropna().unique()
        n_ok = 0
        for raw in unique_smiles:
            if raw in raw_to_canon:
                n_ok += 1
                continue
            result = preprocess_smiles(raw)
            if result is None:
                if raw not in parse_failed:
                    parse_failed.append(raw)
                continue
            canon_iso, canon_flat, mol = result
            raw_to_canon[raw] = canon_iso
            canon_to_mol[canon_iso] = mol
            canon_to_flat[canon_iso] = canon_flat
            n_ok += 1

        logger.info(f"  {ds_name}: {len(unique_smiles)} raw SMILES → "
                     f"{n_ok} canonicalised OK, "
                     f"{len(unique_smiles) - n_ok} parse failures")

    n_raw = len(raw_to_canon)
    n_canon = len(canon_to_mol)
    n_stereo = sum(1 for c in canon_to_mol
                   if canon_to_flat.get(c, c) != c)

    logger.info(f"\n  Phase 1 summary:")
    logger.info(f"    Raw SMILES processed : {n_raw:,}")
    logger.info(f"    Unique canonical (iso): {n_canon:,}")
    logger.info(f"    With stereochemistry  : {n_stereo:,}")
    logger.info(f"    RDKit parse failures  : {len(parse_failed)}")

    return raw_to_canon, canon_to_mol, canon_to_flat, parse_failed, per_dataset


# ═════════════════════════════════════════════════════════════════════
# Cache I/O
# ═════════════════════════════════════════════════════════════════════

def load_cache() -> dict:
    """Load canonical_smiles → iupac_name cache."""
    if not os.path.exists(CACHE_FILE):
        return {}
    df = pd.read_csv(CACHE_FILE)
    cache = {}
    for _, row in df.iterrows():
        smi = str(row.get("canonical_smiles", "")).strip()
        name = str(row.get("iupac_name", "")).strip()
        if smi and name and name != "nan":
            cache[smi] = name
    logger.info(f"  Loaded cache: {len(cache):,} entries from {CACHE_FILE}")
    return cache


def save_cache(cache: dict):
    """Persist cache to CSV."""
    rows = [{"canonical_smiles": s, "iupac_name": n}
            for s, n in sorted(cache.items())]
    for attempt in range(5):
        try:
            pd.DataFrame(rows).to_csv(CACHE_FILE, index=False)
            return
        except PermissionError:
            time.sleep(2)
    logger.error(f"  Could not save cache to {CACHE_FILE} after 5 retries")


def save_failed_resolve(records: list):
    """Save failed collision records."""
    if not records:
        return
    pd.DataFrame(records).to_csv(FAILED_FILE, index=False)
    logger.info(f"  Saved {len(records)} failed-resolve entries to {FAILED_FILE}")


# ═════════════════════════════════════════════════════════════════════
# Phase 2 — API Resolution Cascade
# ═════════════════════════════════════════════════════════════════════

def _is_valid_name(name: str) -> bool:
    """Check that an API-returned name looks like a real IUPAC name."""
    if not name or len(name) > 500:
        return False
    # Reject obvious non-names
    lower = name.lower()
    for bad in ("not found", "n/a", "none", "null", "error", "unknown"):
        if lower == bad:
            return False
    return True


def pubchem_iupac(smiles: str, max_retries: int = 3) -> str | None:
    """PubChem PUG REST: SMILES → IUPAC name (POST for long SMILES)."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                PUBCHEM_POST_URL,
                data={"smiles": smiles},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=20,
            )
            if resp.status_code == 200:
                data = resp.json()
                name = data["PropertyTable"]["Properties"][0].get("IUPACName", "")
                if _is_valid_name(name):
                    return name.strip()
                return None
            elif resp.status_code in (429, 503):
                # Rate-limited or busy → exponential backoff
                wait = min(30, (2 ** attempt) * 2.0)
                logger.debug(f"  PubChem {resp.status_code}, retry in {wait:.0f}s")
                time.sleep(wait)
                continue
            else:
                return None
        except (requests.RequestException, KeyError, IndexError, ValueError):
            if attempt < max_retries - 1:
                time.sleep(1.0)
    return None


def chemspider_iupac(smiles: str) -> str | None:
    """ChemSpider v2: SMILES → IUPAC name."""
    api_key = CHEMSPIDER_API_KEY.strip()
    if not api_key:
        return None

    headers = {"apikey": api_key, "Content-Type": "application/json"}
    try:
        # Step 1 — submit filter
        r = requests.post(CHEMSPIDER_FILTER_URL,
                          json={"smiles": smiles},
                          headers=headers, timeout=20)
        if r.status_code != 200:
            return None
        query_id = r.json().get("queryId")
        if not query_id:
            return None

        # Step 2 — poll for compound ID
        compound_id = None
        for _ in range(6):
            time.sleep(1.0)
            r2 = requests.get(CHEMSPIDER_RESULT_URL.format(query_id),
                              headers=headers, timeout=20)
            if r2.status_code == 200:
                results = r2.json().get("results", [])
                if results:
                    compound_id = results[0]
                    break

        if compound_id is None:
            return None

        # Step 3 — get IUPAC name
        r3 = requests.get(CHEMSPIDER_RECORD_URL.format(compound_id),
                          headers=headers, timeout=20)
        if r3.status_code == 200:
            data = r3.json()
            name = data.get("iupacName") or data.get("IupacName", "")
            if _is_valid_name(name):
                return name.strip()
    except (requests.RequestException, KeyError, ValueError):
        pass
    return None


def nci_cir_iupac(smiles: str) -> str | None:
    """NCI Chemical Identifier Resolver fallback."""
    try:
        url = NCI_CIR_URL.format(quote(smiles, safe=""))
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200 and resp.text.strip():
            name = resp.text.strip().split("\n")[0]
            if _is_valid_name(name):
                return name.strip()
    except (requests.RequestException, Exception):
        pass
    return None


def resolve_all_names(
    unique_canons: list,
    cache: dict,
    limit: int | None = None,
) -> tuple[dict, set]:
    """Resolve IUPAC names for a list of unique canonical SMILES.

    Returns:
        resolved:   dict[canonical_smiles → api_name]
        api_failed: set of canonical SMILES where all 3 APIs failed
    """
    resolved = {}
    api_failed = set()

    # Separate cached from uncached
    uncached = []
    for canon in unique_canons:
        if canon in cache:
            resolved[canon] = cache[canon]
        else:
            uncached.append(canon)

    logger.info(f"  Cache hits: {len(resolved):,} / {len(unique_canons):,}")
    logger.info(f"  To resolve: {len(uncached):,}")

    if not uncached:
        return resolved, api_failed

    if limit is not None:
        uncached = uncached[:limit]
        logger.info(f"  --limit applied: resolving first {len(uncached)}")

    total = len(uncached)
    n_ok = 0
    n_fail = 0
    t_start = time.time()

    try:
        for i, canon in enumerate(uncached):
            # ── Progress ──────────────────────────────────────────
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                pct = 100 * (i + 1) / total
                logger.info(
                    f"  [{pct:5.1f}%] {i+1:>6}/{total:>6} | "
                    f"OK={n_ok:,} fail={n_fail:,} | "
                    f"rate={rate:.1f}/s  ETA={timedelta(seconds=int(eta))}"
                )

            # ── 1. PubChem ────────────────────────────────────────
            name = pubchem_iupac(canon)
            time.sleep(PUBCHEM_DELAY)

            # ── 2. ChemSpider ─────────────────────────────────────
            if name is None:
                name = chemspider_iupac(canon)
                time.sleep(CHEMSPIDER_DELAY)

            # ── 3. NCI CIR ───────────────────────────────────────
            if name is None:
                name = nci_cir_iupac(canon)
                time.sleep(NCI_DELAY)

            # ── Record result ─────────────────────────────────────
            if name:
                resolved[canon] = name
                cache[canon] = name
                n_ok += 1
            else:
                api_failed.add(canon)
                n_fail += 1

            # ── Periodic cache save ───────────────────────────────
            if (n_ok + n_fail) % SAVE_INTERVAL == 0:
                save_cache(cache)

    except KeyboardInterrupt:
        logger.info(f"\n  ⚠ Interrupted — saving cache ({n_ok:,} resolved)...")

    # Final save
    save_cache(cache)

    elapsed = time.time() - t_start
    logger.info(f"\n  Phase 2 complete in {timedelta(seconds=int(elapsed))}")
    logger.info(f"    Resolved: {n_ok:,}")
    logger.info(f"    API-failed: {n_fail:,}")
    logger.info(f"    Cache total: {len(cache):,}")

    return resolved, api_failed


# ═════════════════════════════════════════════════════════════════════
# Phase 3 — Collision Detection & Stereo Resolution
# ═════════════════════════════════════════════════════════════════════

def detect_and_resolve_collisions(
    resolved: dict,
    canon_to_mol: dict,
    canon_to_flat: dict,
) -> tuple[dict, list]:
    """Detect IUPAC name collisions and resolve with stereo prefixes.

    A "collision" = two different canonical isomeric SMILES that
    resolved to the exact same IUPAC name.

    Returns:
        final_names:    dict[canonical_iso → final_unique_name]
        failed_records: list[dict] for failed_resolve.csv
    """
    # ── Invert mapping: name → [list of canonical SMILES] ─────────
    name_to_canons = defaultdict(list)
    for canon, name in resolved.items():
        name_to_canons[name.lower()].append(canon)

    final_names = {}
    failed_records = []
    n_collisions = 0
    n_resolved_collisions = 0

    for name_lower, canons in name_to_canons.items():
        if len(canons) == 1:
            # No collision — keep the original-case name
            final_names[canons[0]] = resolved[canons[0]]
            continue

        n_collisions += 1
        original_name = resolved[canons[0]]   # use first entry's case

        # ── Try stereo prefixes ───────────────────────────────────
        prefix_map = {}
        for canon in canons:
            mol = canon_to_mol.get(canon)
            if mol is None:
                prefix_map[canon] = ""
            else:
                prefix_map[canon] = build_stereo_prefix(mol)

        # Check uniqueness of prefixed names
        proposed = {}
        for canon in canons:
            pname = prefix_map[canon] + original_name
            proposed[canon] = pname

        # Are all proposed names unique?
        name_counts = defaultdict(list)
        for canon, pname in proposed.items():
            name_counts[pname].append(canon)

        all_unique = all(len(v) == 1 for v in name_counts.values())

        if all_unique:
            # Collision resolved!
            n_resolved_collisions += 1
            for canon, pname in proposed.items():
                final_names[canon] = pname
        else:
            # Some still collide — mark conflicting ones as failed,
            # keep the ones that ARE unique.
            for pname, group in name_counts.items():
                if len(group) == 1:
                    final_names[group[0]] = pname
                    n_resolved_collisions += 1
                else:
                    for canon in group:
                        final_names[canon] = ""   # blank
                        flat = canon_to_flat.get(canon, "")
                        failed_records.append({
                            "canonical_smiles": canon,
                            "flat_smiles": flat,
                            "api_name": original_name,
                            "stereo_prefix": prefix_map.get(canon, ""),
                            "reason": "collision_unresolvable",
                        })

    logger.info(f"\n  Phase 3 summary:")
    logger.info(f"    Unique names (no collision)   : "
                f"{sum(1 for v in name_to_canons.values() if len(v) == 1):,}")
    logger.info(f"    Collision groups               : {n_collisions}")
    logger.info(f"    Resolved with stereo prefix    : {n_resolved_collisions}")
    logger.info(f"    Unresolvable (→ blank)         : {len(failed_records)}")

    return final_names, failed_records


# ═════════════════════════════════════════════════════════════════════
# Phase 4 — Apply to Datasets
# ═════════════════════════════════════════════════════════════════════

def apply_names_to_datasets(
    per_dataset: dict,
    raw_to_canon: dict,
    final_names: dict,
    datasets_to_process: dict,
):
    """Write resolved IUPAC names back to each _final.csv.

    Molecules that couldn't be resolved or had collisions get iupac_name = "".
    """
    for ds_name, df in per_dataset.items():
        path = datasets_to_process[ds_name]

        # Map: raw_smiles → canonical_iso → final_iupac_name
        def _resolve(raw):
            canon = raw_to_canon.get(raw)
            if canon is None:
                return ""
            return final_names.get(canon, "")

        df = df.copy()
        df["iupac_name"] = df["smiles"].apply(_resolve)

        # Statistics before drop
        n_total    = len(df)
        n_resolved = (df["iupac_name"] != "").sum()
        n_blank    = n_total - n_resolved

        # ── DROP molecules with no IUPAC name ──────────────────────
        # Keeping blank rows would expose raw SMILES strings to the
        # transformer tokenizer, which only understands IUPAC syntax.
        df = df[df["iupac_name"] != ""].reset_index(drop=True)

        n_unique = df["iupac_name"].nunique()

        # Check for within-dataset IUPAC duplicates
        dupes = df["iupac_name"][df["iupac_name"].duplicated(keep=False)]
        n_dupe_rows = len(dupes)
        n_dupe_names = dupes.nunique()

        logger.info(f"\n  {ds_name}:")
        logger.info(f"    Total molecules       : {n_total:,}")
        logger.info(f"    IUPAC resolved        : {n_resolved:,}  ({100*n_resolved/n_total:.1f}%)")
        logger.info(f"    Dropped (no IUPAC)    : {n_blank:,}")
        logger.info(f"    Final rows            : {len(df):,}")
        logger.info(f"    Unique IUPAC names    : {n_unique:,}")
        if n_dupe_rows > 0:
            logger.info(f"    ⚠ Within-dataset dups : {n_dupe_rows} rows "
                         f"({n_dupe_names} shared names)")

        # Reorder columns and save
        df = df[["smiles", "iupac_name", "is_toxic"]]
        df.to_csv(path, index=False)
        logger.info(f"    Saved → {path}")


# ═════════════════════════════════════════════════════════════════════
# Cross-dataset collision check
# ═════════════════════════════════════════════════════════════════════

def cross_dataset_stats(per_dataset: dict, raw_to_canon: dict, final_names: dict):
    """Report cross-dataset IUPAC name overlaps and conflicts."""
    # Collect (iupac_name, is_toxic) per dataset
    ds_names_labels = {}
    for ds_name, df in per_dataset.items():
        records = {}
        for _, row in df.iterrows():
            canon = raw_to_canon.get(row["smiles"])
            if canon is None:
                continue
            name = final_names.get(canon, "")
            if name:
                records[name] = int(row["is_toxic"])
        ds_names_labels[ds_name] = records

    # Cross-dataset overlaps
    all_names = set()
    for recs in ds_names_labels.values():
        all_names.update(recs.keys())

    n_cross_dup = 0
    n_label_conflict = 0
    for name in all_names:
        in_ds = [ds for ds, recs in ds_names_labels.items() if name in recs]
        if len(in_ds) > 1:
            n_cross_dup += 1
            labels = set(ds_names_labels[ds][name] for ds in in_ds)
            if len(labels) > 1:
                n_label_conflict += 1

    logger.info(f"\n  Cross-dataset:")
    logger.info(f"    Total unique IUPAC names : {len(all_names):,}")
    logger.info(f"    Appear in >1 dataset     : {n_cross_dup:,}")
    logger.info(f"    Label conflicts          : {n_label_conflict:,}")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="STEP 3: SMILES → IUPAC name resolution (V2 rebuild)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Phase 1 only: preprocess and show statistics")
    parser.add_argument("--cache-only", action="store_true",
                        help="Skip API calls, apply existing cache only")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only resolve first N uncached SMILES")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["toxcast", "tox21", "clintox", "herg", "dili"],
                        help="Process one dataset only (default: all five)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing cache, start from scratch")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  STEP 3 — SMILES → IUPAC Name Resolution  (V2)")
    print("=" * 65)

    # ── Which datasets ────────────────────────────────────────────
    if args.dataset:
        datasets_to_process = {args.dataset: DATASETS[args.dataset]}
    else:
        datasets_to_process = DATASETS.copy()

    logger.info(f"Processing: {', '.join(datasets_to_process.keys())}")

    # ══════════════════════════════════════════════════════════════
    # Phase 1 — Preprocessing
    # ══════════════════════════════════════════════════════════════
    logger.info("\n── Phase 1: SMILES Preprocessing ──")

    (raw_to_canon, canon_to_mol, canon_to_flat,
     parse_failed, per_dataset) = collect_and_preprocess(datasets_to_process)

    unique_canons = list(canon_to_mol.keys())

    if args.dry_run:
        # Show estimated API time
        cache = load_cache() if not args.fresh else {}
        n_cached = sum(1 for c in unique_canons if c in cache)
        n_uncached = len(unique_canons) - n_cached
        est_minutes = n_uncached * 0.25 / 60   # ~0.25s per API call average
        print(f"\n  DRY RUN — no API calls or file writes.")
        print(f"  Canonical SMILES: {len(unique_canons):,}")
        print(f"  Already cached  : {n_cached:,}")
        print(f"  Need API calls  : {n_uncached:,}")
        print(f"  Est. time       : ~{est_minutes:.0f} minutes (PubChem only)")
        if parse_failed:
            print(f"\n  Parse failures ({len(parse_failed)}):")
            for s in parse_failed[:10]:
                print(f"    {s[:80]}")
        return

    # ══════════════════════════════════════════════════════════════
    # Phase 2 — API Resolution
    # ══════════════════════════════════════════════════════════════
    logger.info("\n── Phase 2: API Resolution ──")
    logger.info("  Cascade: PubChem → ChemSpider → NCI CIR")

    cache = {} if args.fresh else load_cache()

    if args.cache_only:
        logger.info("  --cache-only: skipping API calls")
        resolved = {}
        api_failed = set()
        for canon in unique_canons:
            if canon in cache:
                resolved[canon] = cache[canon]
            else:
                api_failed.add(canon)
        logger.info(f"  Cache hits: {len(resolved):,}, uncached: {len(api_failed):,}")
    else:
        logger.info("  (Press Ctrl+C to interrupt — progress is auto-saved)\n")
        resolved, api_failed = resolve_all_names(
            unique_canons, cache, limit=args.limit,
        )

    # Add API-failed entries as blank
    for canon in api_failed:
        resolved[canon] = ""

    # ══════════════════════════════════════════════════════════════
    # Phase 3 — Collision Detection & Resolution
    # ══════════════════════════════════════════════════════════════
    logger.info("\n── Phase 3: Collision Resolution ──")

    # Filter to only resolved names (non-blank)
    resolved_only = {k: v for k, v in resolved.items() if v}

    final_names, failed_records = detect_and_resolve_collisions(
        resolved_only, canon_to_mol, canon_to_flat,
    )

    # Merge back API-failed (blank) entries
    for canon in api_failed:
        if canon not in final_names:
            final_names[canon] = ""

    # Also mark parse-failed raw SMILES
    for raw in parse_failed:
        failed_records.append({
            "canonical_smiles": "",
            "flat_smiles": "",
            "api_name": "",
            "stereo_prefix": "",
            "reason": f"rdkit_parse_failed|{raw[:100]}",
        })

    # Save failed resolve log
    if failed_records:
        save_failed_resolve(failed_records)

    # ══════════════════════════════════════════════════════════════
    # Phase 4 — Apply to Datasets
    # ══════════════════════════════════════════════════════════════
    logger.info("\n── Phase 4: Apply to Datasets ──")

    apply_names_to_datasets(
        per_dataset, raw_to_canon, final_names, datasets_to_process,
    )

    # Cross-dataset stats
    cross_dataset_stats(per_dataset, raw_to_canon, final_names)

    # ── Final summary ─────────────────────────────────────────────
    n_final_ok = sum(1 for v in final_names.values() if v)
    # Count ALL blanks: explicit "" entries + SMILES never looked up (absent from dict)
    n_final_blank = len(unique_canons) - n_final_ok
    n_unique = len(set(v for v in final_names.values() if v))

    print("\n" + "=" * 65)
    print(f"  STEP 3 COMPLETE")
    print(f"    Canonical SMILES processed : {len(unique_canons):,}")
    print(f"    IUPAC names assigned       : {n_final_ok:,}")
    print(f"    Left blank (failed/collision): {n_final_blank:,}")
    print(f"    Unique IUPAC names         : {n_unique:,}")
    if failed_records:
        print(f"    Failed resolve log         : {FAILED_FILE}")
    print(f"    Cache                      : {CACHE_FILE}")
    print("=" * 65)
    print("  Next → python steps/step4_verify_lora.py")
    print()


if __name__ == "__main__":
    main()
