from id_unusable.helpers import get_ads_idcs, _log_generic, is_bonded, get_outfile_path, should_write_log
from pathlib import Path
from pymatgen.io.jdftx.outputs import JDFTXOutfile
import numpy as np
from typing import Callable


def surf_of_structure_is_broken(structure, ads_idcs: list[int], bond_scale_factor=1.2):
    if len(ads_idcs):
        if isinstance(ads_idcs[0], list):
            ads_idcs = [idx for sublist in ads_idcs for idx in sublist]
    surf_idcs = [idx for idx in range(len(structure)) if idx not in ads_idcs]
    for surf_idx in surf_idcs:
        other_surfs_idcs = [idx for idx in surf_idcs if idx != surf_idx]
        if not any(is_bonded(structure, surf_idx, idx, scale_factor=bond_scale_factor) for idx in other_surfs_idcs):
            return True
    return False
    

def structure_is_broken(structure, initial_structure, ads_mol: str, bond_scale_factor=1.2):
    ads_idcs = get_ads_idcs(structure, ads_mol)
    if surf_of_structure_is_broken(structure, ads_idcs, bond_scale_factor=bond_scale_factor):
        return True
    if not "_" in ads_mol:
        if len(ads_idcs) < 2:
            return False
        for ads_idx in ads_idcs:
            other_ads_idcs = [idx for idx in ads_idcs if idx != ads_idx]
            initially_bonded_idcs = [idx for idx in other_ads_idcs if is_bonded(initial_structure, ads_idx, idx, scale_factor=bond_scale_factor)]
            currently_bonded_idcs = [idx for idx in other_ads_idcs if is_bonded(structure, ads_idx, idx, scale_factor=bond_scale_factor)]
            if len(initially_bonded_idcs) == 0:
                continue
            else:
                if len(set(initially_bonded_idcs) - set(currently_bonded_idcs)) > 0:
                    return True
        return False
    else:
        ads_idcss = get_ads_idcs(structure, ads_mol)
        for i, ads_idcs in enumerate(ads_idcss):
            # if len(ads_idcs) < 2:
            #     break
            other_ads_idcs = [idx for j, ads_idcs2 in enumerate(ads_idcss) if j != i for idx in ads_idcs2]
            for ads_idx in ads_idcs:
                # Check the other ads_mols to detect if unwanted bonds form between adsorbates
                other_mutual_ads_idcs = [idx for idx in ads_idcs if idx != ads_idx] + other_ads_idcs
                initially_bonded_idcs = [idx for idx in other_mutual_ads_idcs if is_bonded(initial_structure, ads_idx, idx, scale_factor=bond_scale_factor)]
                currently_bonded_idcs = [idx for idx in other_mutual_ads_idcs if is_bonded(structure, ads_idx, idx, scale_factor=bond_scale_factor)]
                if len(initially_bonded_idcs) == 0:
                    continue
                else:
                    if len(set(initially_bonded_idcs) - set(currently_bonded_idcs)) > 0:
                        return True
        return False


def _calc_root_is_broken(calc_root: Path, bond_scale_factor: float):
    checkpath = calc_root / "broken.txt"
    if checkpath.exists():
        pre_bond_scale_factor = float(checkpath.read_text())
        return np.isclose(pre_bond_scale_factor, bond_scale_factor) or pre_bond_scale_factor < bond_scale_factor 
    else:
        return False

def _calc_root_is_not_broken(calc_root: Path, bond_scale_factor: float):
    checkpath = calc_root / "intact.txt"
    if checkpath.exists():
        pre_bond_scale_factor = float(checkpath.read_text())
        return np.isclose(pre_bond_scale_factor, bond_scale_factor) or pre_bond_scale_factor > bond_scale_factor 
    else:
        return False

def _log_is_broken(calc_root: Path, bond_scale_factor: float):
    log_path1 = calc_root / "broken.txt"
    log_path2 = calc_root / "intact.txt"
    _log_generic(log_path1, log_path2, bond_scale_factor)

def _log_is_not_broken(calc_root: Path, bond_scale_factor: float):
    log_path1 = calc_root / "broken.txt"
    log_path2 = calc_root / "intact.txt"
    _log_generic(log_path2, log_path1, bond_scale_factor)


def calc_root_log_shows_broken(calc_root: Path, bond_scale_factor: float) -> bool | None:
    if _calc_root_is_broken(calc_root, bond_scale_factor):
        return True
    if _calc_root_is_not_broken(calc_root, bond_scale_factor):
        return False
    return None

def calc_root_is_broken_operate(calc_root: Path, ads_mol: str, bond_scale_factor=1.4, write_log=True, get_expected_path: Callable | None = None, calc_root_is_finished: Callable | None = None) -> bool | None:
    outfile_path = get_outfile_path(calc_root, get_expected_path=get_expected_path)
    if not outfile_path.exists():
        return None
    outfile = JDFTXOutfile.from_file(outfile_path)
    structure = outfile.structure
    initial_structure = outfile.slices[0].jstrucs[0]
    is_broken = structure_is_broken(structure, initial_structure, ads_mol, bond_scale_factor=bond_scale_factor)
    if write_log:
        if is_broken:
            _log_is_broken(calc_root, bond_scale_factor=bond_scale_factor)
        elif should_write_log(calc_root, calc_root_is_finished):
            _log_is_not_broken(calc_root, bond_scale_factor=bond_scale_factor)
    return is_broken

exception_mols = [
    "O", "H", "N"
]


def calc_root_is_broken(calc_root: Path, ads_mol: str, bond_scale_factor=1.4, check_log=True, write_log=True, test_new_algo: bool = False, get_expected_path: Callable | None = None, calc_root_is_finished: Callable | None = None) -> bool | None:
    if ads_mol in exception_mols:
        return False
    if test_new_algo:
        is_desorbed_log = calc_root_log_shows_broken(calc_root, bond_scale_factor)
        is_desorbed_operate = calc_root_is_broken_operate(calc_root, ads_mol, bond_scale_factor=bond_scale_factor, write_log=False, get_expected_path=get_expected_path, calc_root_is_finished=calc_root_is_finished)
        print(f" Testing for {calc_root}:")
        if is_desorbed_log is None:
            print(is_desorbed_operate)
        elif is_desorbed_log == is_desorbed_operate:
            print(is_desorbed_log)
        else:
            print(f" {is_desorbed_log} --> {is_desorbed_operate}")
        return is_desorbed_operate
    if check_log:
        is_desorbed_log = calc_root_log_shows_broken(calc_root, bond_scale_factor)
        if not is_desorbed_log is None:
            return is_desorbed_log
    return calc_root_is_broken_operate(calc_root, ads_mol, bond_scale_factor=bond_scale_factor, write_log=write_log, get_expected_path=get_expected_path, calc_root_is_finished=calc_root_is_finished)

