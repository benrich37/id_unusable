from pathlib import Path
from ase.data import covalent_radii
import numpy as np
from pymatgen.core.periodic_table import Element
from pymatgen.io.jdftx.outputs import JDFTXOutputs
from pymatgen.io.jdftx.outputs import JDFTXOutfile
from pymatgen.command_line.chargemol_caller import ChargemolAnalysis as _ChargemolAnalysis
from typing import Callable
import warnings
import numpy as np
warnings.filterwarnings("ignore", message="No CHGCAR found. Some properties may be unavailable.")
warnings.filterwarnings("ignore", message="No POTCAR found. Some properties may be unavailable.")
def_cutoff = 0.1

class ChargemolAnalysis(_ChargemolAnalysis):
    def __init__(self, path: Path, run_chargemol=True):
        super().__init__(path=path, run_chargemol=run_chargemol)
        self.bond_matrix = get_bond_order_matrix(cma=self)


def _get_cma_(calc_dir: Path) -> ChargemolAnalysis:
    cma = ChargemolAnalysis(path=calc_dir, run_chargemol=False)
    if cma.structure is None:
        cma.structure = JDFTXOutputs.from_calc_dir(calc_dir).outfile.structure
    return cma

def get_cma(calc_dir: Path) -> ChargemolAnalysis | None:
    try:
        cma = _get_cma_(calc_dir)
        return cma
    except Exception as e:
        return None

def _get_cma(calc_dir: Path | None = None, cma: ChargemolAnalysis | None = None) -> ChargemolAnalysis:
    if cma is None:
        if calc_dir is None:
            raise ValueError("Either calc_dir or cma must be provided.")
        cma = get_cma(calc_dir)
    return cma

def get_bond_order_matrix(calc_dir: Path | None = None, cma: ChargemolAnalysis | None = None) -> np.ndarray:
    cma = _get_cma(calc_dir=calc_dir, cma=cma)
    natoms = len(cma.bond_order_sums)
    bond_matrix = np.zeros((natoms, natoms))
    for idx1, _bo_dict in cma.bond_order_dict.items():
        for bo_dict in _bo_dict["bonded_to"]:
            idx2 = bo_dict["index"]
            val = bo_dict["bond_order"]
            bond_matrix[idx1, idx2] = val
    return bond_matrix


ads_mol_dict = {
    "O": {"O": 1},
    "OH": {"O": 1, "H": 1},
    "NO3": {"N": 1, "O": 3},
    "NO3H": {"N": 1, "O": 3, "H": 1},
    "NO2": {"N": 1, "O": 2},
    "NO2H": {"N": 1, "O": 2, "H": 1},
    "NO": {"N": 1, "O": 1},
    "NOH": {"N": 1, "O": 1, "H": 1},
    "N": {"N": 1},
    "NH": {"N": 1, "H": 1},
    "NH2": {"N": 1, "H": 2},
    "NH3": {"N": 1, "H": 3},
    "H": {"H": 1},
    "surfs": {},
    "clean": {},
}

def get_ads_idcs_singular(structure, ads_mol: str):
    el_type_counts = ads_mol_dict[ads_mol]
    ads_idcs = []
    for el_type, count in el_type_counts.items():
        if structure.composition.get(el_type, 0) < count:
            raise ValueError(f"Structure does not contain enough of element {el_type} for adsorbate {ads_mol}")
        el_idcs = [idx for idx, site in enumerate(structure.sites) if site.species_string == el_type]
        if len(el_idcs) < count:
            raise ValueError(f"Not enough sites of element {el_type} in structure for adsorbate {ads_mol}")
        for i in range(count):
            ads_idcs.append(el_idcs[-(i+1)])
    return ads_idcs

def get_ads_idcs(structure, ads_mol: str) -> list[int] | list[list[int]]:
    if not "_" in ads_mol:
        return get_ads_idcs_singular(structure, ads_mol)
    else:
        ads_idcss = []
        ads_mols = ads_mol.split("_")
        el_type_countss = [ads_mol_dict[ads_mol] for ads_mol in ads_mols]
        all_els = list(set([el for el_type_counts in el_type_countss for el in el_type_counts.keys()]))
        for el in all_els:
            if not el in structure.composition:
                raise ValueError(f"Structure does not contain element {el} required for adsorbate {ads_mol}")
        for el in all_els:
            for el_type_counts in el_type_countss:
                if el not in el_type_counts:
                    el_type_counts[el] = 0
        for i, el_type_counts in enumerate(el_type_countss):
            ads_idcs = []
            upcoming_el_type_counts = {el: sum([el_type_counts[el] for el_type_counts in el_type_countss[i+1:]]) for el in all_els}
            for el_type, count in el_type_counts.items():
                if count == 0:
                    continue
                el_idcs = [idx for idx, site in enumerate(structure.sites) if site.species_string == el_type]
                start_idx = -(count + upcoming_el_type_counts[el_type])
                end_idx = -(upcoming_el_type_counts[el_type]) if upcoming_el_type_counts[el_type] > 0 else None
                # ads_idcs.append(el_idcs[start_idx:end_idx])
                ads_idcs.extend(el_idcs[start_idx:end_idx])
            ads_idcss.append(ads_idcs)
    return ads_idcss


def log_generic(log_path_true, log_path_false, bond_scale_factor: float):
    if log_path_false.exists():
        log_path_false.unlink()
    with open(log_path_true, "w") as f:
        f.write(f"{bond_scale_factor}")

def is_bonded(structure, idx1, idx2, scale_factor=1.2):
    site1 = structure.sites[idx1]
    site2 = structure.sites[idx2]
    dist = np.linalg.norm(site1.coords - site2.coords)
    return dist < scale_factor * (expected_bond_length(structure, idx1, idx2))

def is_bonded_ddec(cma: ChargemolAnalysis, idx1, idx2, cutoff=def_cutoff):
    return cma.bond_matrix[idx1, idx2] > cutoff

def expected_bond_length(structure, idx1, idx2):
    site1 = structure.sites[idx1]
    site2 = structure.sites[idx2]
    el1 = site1.species_string
    el2 = site2.species_string
    r1 = covalent_radii[Element(el1).Z]
    r2 = covalent_radii[Element(el2).Z]
    return r1 + r2

def get_outfile_path(calc_root: Path, get_expected_path: Callable[[Path], Path] | None = None):
    if not get_expected_path is None:
        return get_expected_path(calc_root)
    outfiles = list((calc_root).glob("*out"))
    if len(outfiles) == 0:
        return None
    elif len(outfiles) == 1:
        return outfiles[0]
    else:
        outfile_objects = []
        for outfile in outfiles:
            try:
                outfile_obj = JDFTXOutfile(outfile)
                outfile_objects.append(outfile_obj)
            except Exception as e:
                outfile_objects.append(None)
        valid_outfiles = [outfile for outfile, obj in zip(outfiles, outfile_objects) if not obj is None]
        if not len(valid_outfiles):
            return None
        elif len(valid_outfiles) == 1:
            return valid_outfiles[0]
        else:
            outfiles_by_last_modification = sorted(valid_outfiles, key=lambda f: f.stat().st_mtime, reverse=True)
            return outfiles_by_last_modification[0]
        
def should_write_log(calc_root: Path, calc_root_is_finished: Callable[[Path], bool] | None = None):
    if calc_root_is_finished is None:
        return True
    else:
        return calc_root_is_finished(calc_root)