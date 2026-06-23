
from id_unusable.helpers import get_ads_idcs, _log_generic, is_bonded, get_cma, def_cutoff, ChargemolAnalysis, is_bonded_ddec, get_outfile_path, should_write_log
from pathlib import Path
from pymatgen.io.jdftx.outputs import JDFTXOutfile
import numpy as np
from typing import Callable


def get_surf_idcs(structure, ads_idcs):
    surf_idcs = [idx for idx in range(len(structure)) if idx not in ads_idcs]
    return surf_idcs

def get_closest_surf_idx(structure, ads_idcs, ignore_idcs: list[int] | None = None):
    if ignore_idcs is None:
        ignore_idcs = []
    ads_coords = np.array([structure.sites[idx].coords for idx in ads_idcs])
    surf_idcs = get_surf_idcs(structure, ads_idcs)
    surf_idcs = [idx for idx in surf_idcs if idx not in ignore_idcs]
    surf_coords = np.array([structure.sites[idx].coords for idx in surf_idcs])
    dists = np.zeros((len(ads_idcs), len(surf_idcs)))
    for i in range(len(ads_idcs)):
        dists[i] = np.linalg.norm(surf_coords - ads_coords[i], axis=1)
    ads_idx, surf_idx = np.unravel_index(np.argmin(dists), dists.shape)
    return surf_idcs[surf_idx], ads_idcs[ads_idx]



def structure_is_desorbed(structure, ads_mol: str, bond_scale_factor=1.2):
    if not "_" in ads_mol:
        ads_idcs = get_ads_idcs(structure, ads_mol)
        surf_idx, ads_idx = get_closest_surf_idx(structure, ads_idcs)
        return not is_bonded(structure, surf_idx, ads_idx, scale_factor=bond_scale_factor)
    else:
        ads_idcss = get_ads_idcs(structure, ads_mol)
        for i, ads_idcs in enumerate(ads_idcss):
            other_ads_idcs = [idx for j, ads_idcs2 in enumerate(ads_idcss) if j != i for idx in ads_idcs2]
            surf_idx, ads_idx = get_closest_surf_idx(structure, ads_idcs, ignore_idcs=other_ads_idcs)
            if not is_bonded(structure, surf_idx, ads_idx, scale_factor=bond_scale_factor):
                return True
        return False

def cma_is_desorbed(cma: ChargemolAnalysis, ads_mol: str, cutoff=def_cutoff):
    if not "_" in ads_mol:
        ads_idcs = get_ads_idcs(cma.structure, ads_mol)
        surf_idcs = get_surf_idcs(cma.structure, ads_idcs)
        for ads_idx in ads_idcs:
            if any(is_bonded_ddec(cma, ads_idx, surf_idx, cutoff=cutoff) for surf_idx in surf_idcs):
                return False
        return True
    else:
        ads_idcss = get_ads_idcs(cma.structure, ads_mol)
        for i, ads_idcs in enumerate(ads_idcss):
            other_ads_idcs = [idx for j, ads_idcs2 in enumerate(ads_idcss) if j != i for idx in ads_idcs2]
            surf_idcs = get_surf_idcs(cma.structure, ads_idcs)
            surf_idcs = [idx for idx in surf_idcs if idx not in other_ads_idcs]
            for ads_idx in ads_idcs:
                if any(is_bonded_ddec(cma, ads_idx, surf_idx, cutoff=cutoff) for surf_idx in surf_idcs):
                    break
            return True
        return False



def _calc_root_is_desorbed(calc_root: Path, bond_scale_factor: float):
    checkpath = calc_root / "desorbed.txt"
    if checkpath.exists():
        pre_bond_scale_factor = float(checkpath.read_text())
        return np.isclose(pre_bond_scale_factor, bond_scale_factor) or pre_bond_scale_factor < bond_scale_factor 
    else:
        return False

def _calc_root_is_not_desorbed(calc_root: Path, bond_scale_factor: float):
    checkpath = calc_root / "adsorbed.txt"
    if checkpath.exists():
        pre_bond_scale_factor = float(checkpath.read_text())
        return np.isclose(pre_bond_scale_factor, bond_scale_factor) or pre_bond_scale_factor > bond_scale_factor 
    else:
        return False

def _log_is_desorbed(calc_root: Path, bond_scale_factor: float):
    log_path1 = calc_root / "desorbed.txt"
    log_path2 = calc_root / "adsorbed.txt"
    _log_generic(log_path1, log_path2, bond_scale_factor)

def _log_is_not_desorbed(calc_root: Path, bond_scale_factor: float):
    log_path1 = calc_root / "desorbed.txt"
    log_path2 = calc_root / "adsorbed.txt"
    _log_generic(log_path2, log_path1, bond_scale_factor)


###

def _calc_root_is_desorbed_cma(calc_root: Path, cutoff: float):
    checkpath = calc_root / "desorbed_cma.txt"
    if checkpath.exists():
        pre_cutoff = float(checkpath.read_text())
        return np.isclose(pre_cutoff, cutoff) or pre_cutoff > cutoff 
    else:
        return False

def _calc_root_is_not_desorbed_cma(calc_root: Path, cutoff: float):
    checkpath = calc_root / "adsorbed_cma.txt"
    if checkpath.exists():
        pre_cutoff = float(checkpath.read_text())
        return np.isclose(pre_cutoff, cutoff) or pre_cutoff < cutoff 
    else:
        return False

def _log_is_desorbed_cma(calc_root: Path, cutoff: float):
    log_path1 = calc_root / "desorbed_cma.txt"
    log_path2 = calc_root / "adsorbed_cma.txt"
    _log_generic(log_path1, log_path2, cutoff)

def _log_is_not_desorbed_cma(calc_root: Path, cutoff: float):
    log_path1 = calc_root / "desorbed_cma.txt"
    log_path2 = calc_root / "adsorbed_cma.txt"
    _log_generic(log_path2, log_path1, cutoff)

###


def calc_root_log_shows_desorbed(calc_root: Path, bond_scale_factor: float) -> bool | None:
    if _calc_root_is_desorbed(calc_root, bond_scale_factor):
        return True
    if _calc_root_is_not_desorbed(calc_root, bond_scale_factor):
        return False
    return None

def calc_root_log_shows_desorbed_cma(calc_root: Path, cutoff: float) -> bool | None:
    if _calc_root_is_desorbed_cma(calc_root, cutoff):
        return True
    if _calc_root_is_not_desorbed_cma(calc_root, cutoff):
        return False
    return None


def calc_root_is_desorbed_cov_radii(
        calc_root: Path, ads_mol: str, 
        bond_scale_factor=1.2, 
        check_log=True, write_log=True, 
        get_expected_path: Callable[[Path], Path] | None = None, 
        calc_root_is_finished: Callable[[Path], bool] | None = None
        ) -> bool:
    if check_log:
        is_desorbed_log = calc_root_log_shows_desorbed(calc_root, bond_scale_factor)
        if not is_desorbed_log is None:
            return is_desorbed_log
    outfile_path = get_outfile_path(calc_root, get_expected_path=get_expected_path)
    if outfile_path is None:
        return None
    elif not outfile_path.exists():
        return None
    outfile = JDFTXOutfile.from_file(outfile_path)
    structure = outfile.structure
    is_desorbed = structure_is_desorbed(structure, ads_mol, bond_scale_factor=bond_scale_factor)
    if write_log:
        if is_desorbed:
            _log_is_desorbed(calc_root, bond_scale_factor=bond_scale_factor)
        elif should_write_log(calc_root, calc_root_is_finished):
            _log_is_not_desorbed(calc_root, bond_scale_factor=bond_scale_factor)
    return is_desorbed

def calc_root_is_desorbed_cma(
        calc_root: Path, ads_mol: str, 
        cutoff=def_cutoff, 
        check_log=True, write_log=True, 
        get_expected_path: Callable[[Path], Path] | None = None, 
        calc_root_is_finished: Callable[[Path], bool] | None = None
        ) -> bool | None:
    if check_log:
        is_desorbed_log = calc_root_log_shows_desorbed_cma(calc_root, cutoff)
        if not is_desorbed_log is None:
            return is_desorbed_log
    outfile_path = get_outfile_path(calc_root, get_expected_path=get_expected_path)
    if outfile_path is None:
        return None
    elif not outfile_path.exists():
        return None
    cma = get_cma(outfile_path.parent)
    if cma is None:
        return None
    is_desorbed = cma_is_desorbed(cma, ads_mol, cutoff=cutoff)
    if write_log:
        if is_desorbed:
            _log_is_desorbed_cma(calc_root, cutoff=cutoff)
        elif should_write_log(calc_root, calc_root_is_finished):
            _log_is_not_desorbed_cma(calc_root, cutoff=cutoff)
    return is_desorbed


def calc_root_is_desorbed(
        calc_root: Path, ads_mol: str, 
        bond_scale_factor=1.2, cutoff=def_cutoff, 
        check_log=True, write_log=True, 
        get_expected_path: Callable[[Path], Path] | None = None, 
        calc_root_is_finished: Callable[[Path], bool] | None = None
        ) -> bool | None:
    is_desorbed_cov = calc_root_is_desorbed_cov_radii(calc_root, ads_mol, bond_scale_factor=bond_scale_factor, check_log=check_log, write_log=write_log, get_expected_path=get_expected_path, calc_root_is_finished=calc_root_is_finished)
    if not is_desorbed_cov:
        is_desorbed_cma = calc_root_is_desorbed_cma(calc_root, ads_mol, cutoff=cutoff, check_log=check_log, write_log=write_log, get_expected_path=get_expected_path, calc_root_is_finished=calc_root_is_finished)
        if is_desorbed_cma is None:
            return is_desorbed_cov
        else:
            return is_desorbed_cma
    else:
        return is_desorbed_cov