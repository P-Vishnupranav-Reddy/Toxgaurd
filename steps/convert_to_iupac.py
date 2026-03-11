#!/usr/bin/env python3
"""
Convert all non-IUPAC trivial/common names in build_common_molecules.py
to systematic IUPAC names using PubChem REST API + manual overrides.

Usage:
    python steps/convert_to_iupac.py                 # Dry run (show changes)
    python steps/convert_to_iupac.py --apply          # Apply changes to .py file
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE  = os.path.dirname(SCRIPT_DIR)
BUILD_PY   = os.path.join(WORKSPACE, "steps", "build_common_molecules.py")
CACHE_FILE = os.path.join(WORKSPACE, "steps", "pubchem_iupac_cache.json")
MAPPING_FILE = os.path.join(WORKSPACE, "steps", "iupac_mapping.json")

# ---------------------------------------------------------------------------
# MANUAL OVERRIDES
# PubChem returns retained/trivial names for these; we force systematic IUPAC.
# Key: current name in file (lowercase matched).  Value: desired output name.
# If value == key, entry is kept as-is (prevents PubChem regression).
# ---------------------------------------------------------------------------
MANUAL_OVERRIDES = {
    # --- Simple retained organic → systematic ---
    "phenol":        "hydroxybenzene",
    "aniline":       "benzenamine",
    "styrene":       "ethenylbenzene",
    "benzaldehyde":  "benzenecarbaldehyde",
    "urea":          "carbonyl diamide",
    "acetonitrile":  "ethanenitrile",
    "formamide":     "methanamide",
    "acetamide":     "ethanamide",

    # --- Already-systematic names PubChem reverts to trivial → KEEP ---
    "ethyne":              "ethyne",              # PubChem: acetylene
    "trichloromethane":    "trichloromethane",     # PubChem: chloroform
    "tetrachloromethane":  "tetrachloromethane",   # PubChem: might return carbon tetrachloride
    "1,2-dimethylbenzene": "1,2-dimethylbenzene",  # PubChem: 1,2-xylene
    "1,3-dimethylbenzene": "1,3-dimethylbenzene",
    "1,4-dimethylbenzene": "1,4-dimethylbenzene",

    # --- Esters: keep systematic -methanoate / -ethanoate forms ---
    "methyl methanoate":  "methyl methanoate",    # PubChem: methyl formate
    "methyl ethanoate":   "methyl ethanoate",     # PubChem: methyl acetate
    "ethyl ethanoate":    "ethyl ethanoate",      # PubChem: ethyl acetate
    "propyl ethanoate":   "propyl ethanoate",
    "butyl ethanoate":    "butyl ethanoate",
    "pentyl ethanoate":   "pentyl ethanoate",
    "methyl propanoate":  "methyl propanoate",
    "ethyl propanoate":   "ethyl propanoate",
    "methyl butanoate":   "methyl butanoate",
    "ethyl butanoate":    "ethyl butanoate",
    "phenyl ethanoate":   "phenyl ethanoate",
    "benzyl ethanoate":   "benzyl ethanoate",

    # --- IUPAC-retained heterocyclic / fused ring names → KEEP ---
    "naphthalene":   "naphthalene",
    "azulene":       "azulene",
    "thiophene":     "thiophene",
    "furan":         "furan",
    "pyridine":      "pyridine",
    "oxidane":       "oxidane",
    "benzene":       "benzene",
    "anthracene":    "anthracene",
    "phenanthrene":  "phenanthrene",
    "pyrene":        "pyrene",
    "chrysene":      "chrysene",
    "triphenylene":  "triphenylene",
    "acenaphthylene":"acenaphthylene",
    "fluoranthene":  "fluoranthene",

    # --- Inorganic: keep current reasonable names (PubChem uses odd formats) ---
    "hydrochloric acid":  "hydrochloric acid",   # PubChem: chlorane
    "hydrobromic acid":   "hydrobromic acid",    # PubChem: bromane
    "hydrofluoric acid":  "hydrofluoric acid",   # PubChem: fluorane
    "sulfuric acid":      "sulfuric acid",
    "nitric acid":        "nitric acid",
    "phosphoric acid":    "phosphoric acid",
    "perchloric acid":    "perchloric acid",
    "carbonic acid":      "carbonic acid",
    "hydrogen chloride":  "hydrogen chloride",   # PubChem: chlorane
    "hydrogen bromide":   "hydrogen bromide",
    "hydrogen fluoride":  "hydrogen fluoride",
    "nitrogen":           "nitrogen",            # PubChem: molecular nitrogen
    "oxygen":             "oxygen",
    "fluorine":           "fluorine",
    "chlorine":           "chlorine",
    "bromine":            "bromine",
    "ammonia":            "ammonia",             # PubChem: azane (valid but unusual)
    "sodium chloride":    "sodium chloride",
    "potassium chloride": "potassium chloride",
    "sodium bromide":     "sodium bromide",
    "potassium bromide":  "potassium bromide",
    "sodium iodide":      "sodium iodide",
    "sodium fluoride":    "sodium fluoride",
    "sodium carbonate":   "sodium carbonate",    # PubChem: disodium;carbonate
    "potassium carbonate":"potassium carbonate",
    "sodium hydrogen carbonate": "sodium hydrogen carbonate",
    "calcium carbonate":  "calcium carbonate",
    "calcium oxide":      "calcium oxide",       # PubChem: oxocalcium
    "calcium hydroxide":  "calcium hydroxide",
    "magnesium oxide":    "magnesium oxide",     # PubChem: oxomagnesium
    "magnesium hydroxide":"magnesium hydroxide",
    "zinc oxide":         "zinc oxide",          # PubChem: zinc oxygen(2-)
    "zinc sulfate":       "zinc sulfate",
    "silicon dioxide":    "silicon dioxide",     # PubChem: dioxosilane
    "titanium dioxide":   "titanium dioxide",
    "aluminum oxide":     "aluminium oxide",
    "iron oxide":         "iron(III) oxide",
    "barium sulfate":     "barium sulfate",      # PubChem: barium(2+) sulfate
    "bismuth oxide":      "bismuth(III) oxide",
    "arsenic trioxide":   "arsenic trioxide",    # PubChem: semicolon format
    "antimony trioxide":  "antimony trioxide",
    "ammonium nitrate":   "ammonium nitrate",    # PubChem: azanium nitrate
    "carbon dioxide":     "carbon dioxide",
    "carbon monoxide":    "carbon monoxide",
    "sulfur dioxide":     "sulfur dioxide",
    "sulfur trioxide":    "sulfur trioxide",
    "nitrogen dioxide":   "nitrogen dioxide",
    "dinitrogen tetroxide":"dinitrogen tetroxide",
    "dinitrogen oxide":   "dinitrogen oxide",
    "hydrogen peroxide":  "hydrogen peroxide",
    "sodium peroxide":    "sodium peroxide",
    "hydrogen sulfide":   "hydrogen sulfide",    # PubChem: sulfane
    "hydrogen cyanide":   "hydrogen cyanide",    # PubChem: formonitrile
    "ozone":              "ozone",

    # --- Inorganic oxidation-state fixes ---
    "iron sulfate":     "iron(II) sulfate",
    "iron chloride":    "iron(III) chloride",
    "lead oxide":       "lead(II) oxide",
    "mercury chloride": "mercury(II) chloride",
    "nickel chloride":  "nickel(II) chloride",
    "cobalt chloride":  "cobalt(II) chloride",
    "cobalt oxide":     "cobalt(II,III) oxide",
    "nickel oxide":     "nickel(II) oxide",
    "cadmium oxide":    "cadmium(II) oxide",
    "mercury oxide":    "mercury(II) oxide",
    "tin chloride":     "tin(II) chloride",
    "lead chloride":    "lead(II) chloride",

    # --- Sulfur compounds: keep current systematic ---
    "carbon disulfide":  "carbon disulfide",     # PubChem: methanedithione
    "dimethyl sulfide":  "dimethyl sulfide",     # PubChem: methylsulfanylmethane
    "dimethyl sulfoxide":"dimethyl sulfoxide",    # PubChem: methylsulfinylmethane
    "dimethyl sulfone":  "dimethyl sulfone",      # PubChem: methylsulfonylmethane
    "dimethyl sulfate":  "dimethyl sulfate",
    "diethyl sulfate":   "diethyl sulfate",

    # --- More inorganic: PubChem returns odd additive nomenclature ---
    "chromic acid":        "chromic acid",         # PubChem: dihydroxy(dioxo)chromium
    "beryllium oxide":     "beryllium oxide",      # PubChem: oxoberyllium
    "osmium tetroxide":    "osmium tetroxide",     # PubChem: tetraoxoosmium
    "lead sulfide":        "lead(II) sulfide",     # PubChem: sulfanylidenelead
    "manganese dioxide":   "manganese dioxide",    # PubChem: dioxomanganese
    "vanadium pentoxide":  "vanadium pentoxide",   # PubChem: very long additive form
    "methyl isothiocyanate":"methyl isothiocyanate",# PubChem: odd additive form
    "zinc chloride":       "zinc chloride",        # PubChem: dichlorozinc
    "cadmium chloride":    "cadmium chloride",     # PubChem: dichlorocadmium
    "thallium chloride":   "thallium(I) chloride", # PubChem: chlorothallium
    "ammonium fluoride":   "ammonium fluoride",    # PubChem: azanium fluoride
    "chromium trioxide":   "chromium trioxide",    # PubChem: trioxochromium

    # --- Not found on PubChem: provide names manually ---
    "lead acetate":        "lead(II) acetate",
    "arsenic pentoxide":   "diarsenic pentoxide",
}

# ---------------------------------------------------------------------------
# ENTRIES TO REMOVE  (not single molecules / IUPAC impractically long)
# ---------------------------------------------------------------------------
REMOVE_SET = {
    "insulin",              # protein — IUPAC is ~2000 chars
    "calamine",             # mineral mixture (ZnO + Fe2O3)
    "heparin",              # polysaccharide polymer
    "cholestyramine",       # polymeric resin
    "oleum",                # fuming H2SO4 mixture, not a discrete compound
    "nitroprusside",        # complex metal coordination compound
    "ivermectin",           # extremely complex macrolide
    "sucralfate",           # aluminum complex salt
    "butylated hydroxyanisole", # mixture of isomers, not a single compound
    "aflatoxin",            # generic class name (B1/B2/G1/G2)
    "ochratoxin",           # generic class name
    "fumonisin",            # generic class name
}

# ---------------------------------------------------------------------------
# TRIVIAL-NAME BLACKLIST
# If PubChem returns one of these for a name we already have in more-systematic
# form, we KEEP the current name to prevent regression.
# ---------------------------------------------------------------------------
TRIVIAL_BLACKLIST = {
    # Common trivial names
    "toluene", "xylene", "o-xylene", "m-xylene", "p-xylene",
    "1,2-xylene", "1,3-xylene", "1,4-xylene",
    "cumene", "cymene", "mesitylene", "durene",
    "acetone", "acrolein", "acetylene",
    "diethyl ether", "dimethyl ether",
    "anisole", "phenetole",
    "formaldehyde", "acetaldehyde", "benzaldehyde",
    "chloroform", "carbon tetrachloride",
    # Trivial acid names
    "formic acid", "acetic acid", "propionic acid", "butyric acid",
    "valeric acid", "caproic acid", "caprylic acid", "capric acid",
    "lauric acid", "myristic acid", "palmitic acid", "stearic acid",
    "oleic acid", "linoleic acid",
    "oxalic acid", "malonic acid", "succinic acid", "glutaric acid",
    "adipic acid", "pimelic acid", "suberic acid",
    # Trivial ester names
    "methyl formate", "ethyl formate", "methyl acetate", "ethyl acetate",
    "propyl acetate", "butyl acetate", "pentyl acetate", "benzyl acetate",
    # Other
    "glycerol", "glycol", "ethylene glycol", "propylene glycol",
    "catechol", "resorcinol", "hydroquinone",
    "phenol", "aniline", "styrene", "urea",
    "acetonitrile", "formamide", "acetamide",
    "adrenaline", "epinephrine",
    # PubChem unusual inorganic formats
    "chlorane", "bromane", "fluorane", "sulfane", "azane",
    "molecular nitrogen", "molecular oxygen", "molecular fluorine",
    "molecular chlorine", "molecular bromine",
    "formonitrile",
}

# ---------------------------------------------------------------------------
# MAX IUPAC NAME LENGTH — if PubChem IUPAC exceeds this, REMOVE the entry
# (e.g. proteins, massive polymers)
# ---------------------------------------------------------------------------
MAX_IUPAC_LENGTH = 500

# ---------------------------------------------------------------------------
# PubChem API
# ---------------------------------------------------------------------------
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def query_pubchem(name, retries=3):
    """Query PubChem REST API for the IUPAC Name of a compound."""
    encoded = urllib.parse.quote(name, safe="")
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded}/property/IUPACName/JSON"
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ToxGuard/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["PropertyTable"]["Properties"][0]["IUPACName"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                     # compound not found
            if attempt < retries - 1:
                time.sleep(1)
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_candidates(path):
    """Extract (name, label) pairs from the CANDIDATES list in a .py file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return re.findall(r'\("([^"]+)",\s*(\d)\)', content)


def systematic_score(name):
    """Heuristic score — higher = more likely to be systematic IUPAC."""
    score = 0
    if re.search(r"\d", name):
        score += 3                             # numbered locants
    score += name.count("-") * 0.5             # hyphens
    score += name.count(",") * 0.3             # comma locants
    if re.search(r"[\(\)\[\]]", name):
        score += 1                             # brackets
    score += len(name) / 80                    # length (systematic tends longer)
    return score


def should_keep_current(current, pubchem_name):
    """Return True if current name is more systematic than PubChem result."""
    if current.lower() == pubchem_name.lower():
        return True
    if pubchem_name.lower() in {t.lower() for t in TRIVIAL_BLACKLIST}:
        return True
    # Reject PubChem semicolon-separated ionic notation (e.g. "disodium;carbonate")
    if ";" in pubchem_name:
        return True
    # Reject PubChem "oxygen(2-)" style ionic notation
    if re.search(r"\(\d[+-]\)", pubchem_name):
        return True
    # Current has systematic features but PubChem returned a simpler trivial name
    if systematic_score(current) > systematic_score(pubchem_name) + 0.5:
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert non-IUPAC names to systematic IUPAC via PubChem"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Apply changes to build_common_molecules.py")
    args = parser.parse_args()

    # 1. Extract all names -------------------------------------------------
    candidates = extract_candidates(BUILD_PY)
    seen = set()
    unique = []
    for name, label in candidates:
        key = name.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(name.strip())

    total = len(unique)
    print(f"Total unique CANDIDATE names: {total}\n")

    # 2. Load PubChem cache ------------------------------------------------
    cache = load_cache()

    # 3. Process each name -------------------------------------------------
    changes   = {}          # old -> new
    removals  = set()       # names to delete
    unchanged = []
    errors    = []

    override_lower = {k.lower(): v for k, v in MANUAL_OVERRIDES.items()}
    remove_lower   = {r.lower() for r in REMOVE_SET}

    for i, name in enumerate(unique):
        tag = f"[{i+1:4d}/{total}]"
        low = name.lower()

        # --- Manual override ---
        if low in override_lower:
            new = override_lower[low]
            if new.lower() != low:
                changes[name] = new
                print(f"{tag} OVERRIDE  {name!r:40s} -> {new!r}")
            # else: same name (keep as-is entry in overrides)
            continue

        # --- Removal ---
        if low in remove_lower:
            removals.add(name)
            print(f"{tag} REMOVE    {name!r}")
            continue

        # --- PubChem lookup ---
        cache_key = low
        if cache_key in cache:
            iupac = cache[cache_key]
        else:
            iupac = query_pubchem(name)
            cache[cache_key] = iupac
            save_cache(cache)           # incremental save
            time.sleep(0.12)            # rate-limit (~8 req/s)

        if iupac is None:
            errors.append(name)
            print(f"{tag} NOT FOUND {name!r}")
            continue

        # --- Compare ---
        if iupac.lower() == low:
            unchanged.append(name)
            continue

        # Prevent PubChem from reverting systematic → trivial
        if should_keep_current(name, iupac):
            unchanged.append(name)
            continue

        # Reject absurdly long IUPAC names (proteins, etc.)
        if len(iupac) > MAX_IUPAC_LENGTH:
            removals.add(name)
            print(f"{tag} REMOVE    {name!r}  (IUPAC {len(iupac)} chars)")
            continue

        changes[name] = iupac
        print(f"{tag} CHANGE    {name!r:40s} -> {iupac!r}")

    # 4. Summary -----------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  Unchanged : {len(unchanged):5d}")
    print(f"  Changes   : {len(changes):5d}")
    print(f"  Removals  : {len(removals):5d}")
    print(f"  Not found : {len(errors):5d}")
    print(f"{'=' * 70}")

    if errors:
        print("\nNot found on PubChem:")
        for e in errors:
            print(f"    {e}")

    # 5. Save mapping ------------------------------------------------------
    mapping = {
        "changes":  changes,
        "removals": sorted(removals),
        "errors":   errors,
    }
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\nMapping saved -> {MAPPING_FILE}")

    # 6. Apply (optional) --------------------------------------------------
    if not args.apply:
        print("\nDry run complete.  Re-run with --apply to modify the .py file.")
        return

    print("\nApplying changes to build_common_molecules.py ...")
    with open(BUILD_PY, "r", encoding="utf-8") as f:
        content = f.read()

    applied_changes = 0
    applied_removals = 0

    # 6a. Name replacements
    for old_name, new_name in changes.items():
        old_pat = f'("{old_name}",'
        new_pat = f'("{new_name}",'
        if old_pat in content:
            content = content.replace(old_pat, new_pat)
            applied_changes += 1
        else:
            print(f"  WARNING: could not find {old_pat!r}")

    # 6b. Line removals
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        skip = False
        for name in removals:
            if re.match(r'\s*\("' + re.escape(name) + r'",', line):
                skip = True
                applied_removals += 1
                break
        if not skip:
            new_lines.append(line)
    content = "\n".join(new_lines)

    with open(BUILD_PY, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  Applied {applied_changes} name changes, {applied_removals} removals.")
    print("Done.")


if __name__ == "__main__":
    main()
