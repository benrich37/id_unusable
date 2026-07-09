from id_unusable.helpers import get_ads_idcs, log_generic, is_bonded, get_outfile_path, should_write_log
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
            print(f"surf of structure of composition {structure.composition} is broken: surf_idx={surf_idx} ({structure[surf_idx]})has no bonds to other surf atoms")
            return True
    return False
    
def structure_is_broken_singular(structure, initial_structure, ads_idcs: list[int], bond_scale_factor=1.2):
    if len(ads_idcs) < 2:
        return False
    for ads_idx in ads_idcs:
        other_ads_idcs = [idx for idx in ads_idcs if idx != ads_idx]
        if len(other_ads_idcs) and (not any(is_bonded(structure, ads_idx, idx, scale_factor=bond_scale_factor) for idx in other_ads_idcs)):
            print(f"structure of composition {structure.composition} is broken: ads_idx={ads_idx} ({structure[ads_idx]}) has no bonds to other ads atoms")
            return True
        initially_bonded_idcs = [idx for idx in other_ads_idcs if is_bonded(initial_structure, ads_idx, idx, scale_factor=bond_scale_factor)]
        currently_bonded_idcs = [idx for idx in other_ads_idcs if is_bonded(structure, ads_idx, idx, scale_factor=bond_scale_factor)]
        if len(initially_bonded_idcs) == 0:
            continue
        else:
            if len(set(initially_bonded_idcs) - set(currently_bonded_idcs)) > 0:
                print(f"structure of composition {structure.composition} is broken: ads_idx={ads_idx} ({structure[ads_idx]}) has lost bonds to other ads atoms")
                print(f"  initially_bonded_idcs={initially_bonded_idcs}")
                print(f"  currently_bonded_idcs={currently_bonded_idcs}")
                return True
    return False


from id_unusable.helpers import get_super_ads_mol_count_dict, get_ads_idcs_singular_from_el_type_counts, ads_mol_dict


def is_matching_el_type_counts(el_type_counts1: dict[str, int], el_type_counts2: dict[str, int]) -> bool:
    if len(el_type_counts1) != len(el_type_counts2):
        return False
    for el_type, count in el_type_counts1.items():
        if not el_type in el_type_counts2:
            return False
        if el_type_counts2[el_type] != count:
            return False
    return True


def ads_idcss_matches_ads_mol_str(ads_idcss: list[list[int]], ads_mol: str, structure) -> bool:
    if not "_" in ads_mol:
        raise ValueError("compare_ads_idcss_to_ads_mol_str is only for multi-adsorbate molecules")
    el_type_countss = [ads_mol_dict[ads_mol].copy() for ads_mol in ads_mol.split("_")]
    for ads_idcs in ads_idcss:
        el_type_counts = {}
        for idx in ads_idcs:
            el_type = structure.sites[idx].species_string
            if el_type in el_type_counts:
                el_type_counts[el_type] += 1
            else:
                el_type_counts[el_type] = 1
        match_etc_idcs = [i for i, etc in enumerate(el_type_countss) if is_matching_el_type_counts(etc, el_type_counts)]
        if not len(match_etc_idcs):
            return False
        else:
            el_type_countss.pop(match_etc_idcs[0])
    if not len(el_type_countss):
        return True
    return False

def get_ads_idcss_from_trusted_structure(trusted_structure, ads_mol: str, bond_scale_factor=1.2):
    assert "_" in ads_mol, "get_ads_idcss_from_trusted_structure is only for multi-adsorbate molecules"
    el_type_counts = get_super_ads_mol_count_dict(ads_mol)
    all_ads_idcs = get_ads_idcs_singular_from_el_type_counts(trusted_structure, el_type_counts)
    ads_idcss: list[set] = []
    for ads_idx in all_ads_idcs:
        containing_ads_idx = [i for i, ads_idcs in enumerate(ads_idcss) if ads_idx in ads_idcs]
        if len(containing_ads_idx) > 1:
            combine_sets = [ads_idcss[i] for i in containing_ads_idx]
            super_list = []
            for s in combine_sets:
                super_list.extend(list(s))
            super_set = set(super_list)
            ads_idcss = [ads_idcs for i, ads_idcs in enumerate(ads_idcss) if i not in containing_ads_idx]
            # ads_idcss.append(super_set)
            containing_ads_idx = [i for i, ads_idcs in enumerate(ads_idcss) if ads_idx in ads_idcs]
            ads_idcs = list(super_set)
        if len(containing_ads_idx) == 0:
            ads_idcs = [ads_idx]
        else:
            ads_idcs = list(ads_idcss[containing_ads_idx[0]])
            ads_idcs.append(ads_idx)
            ads_idcss = [ads_idcs for i, ads_idcs in enumerate(ads_idcss) if i not in containing_ads_idx]
        other_ads_idcs = [idx for idx in all_ads_idcs if idx != ads_idx]
        bonded_to_ads_idx = [idx for idx in other_ads_idcs if is_bonded(trusted_structure, ads_idx, idx, scale_factor=bond_scale_factor)]
        ads_idcs.extend(bonded_to_ads_idx)
        ads_idcss.append(set(ads_idcs))
    n_mols = len(ads_mol.split("_"))
    if n_mols != len(ads_idcss):
        raise ValueError(f"Number of adsorbate molecules in ads_mol ({n_mols}) does not match number of adsorbate index sets found ({len(ads_idcss)})")
    if not ads_idcss_matches_ads_mol_str(ads_idcss, ads_mol, trusted_structure):
        raise ValueError(f"Adsorbate index sets found do not match ads_mol string: {ads_idcss} vs {ads_mol}")
    return ads_idcss

def structure_is_broken_multi(structure, initial_structure, ads_idcss: list[list[int]], ads_mol, bond_scale_factor=1.2, trust_initial_structure = False):
    return structure_is_broken_multi_checker(structure, initial_structure, ads_idcss, bond_scale_factor=bond_scale_factor, trust_initial_structure=trust_initial_structure)
    # broken = structure_is_broken_multi_checker(structure, initial_structure, ads_idcss, bond_scale_factor=bond_scale_factor, trust_initial_structure=trust_initial_structure)
    # if broken and retry:
    #     ads_idcss = get_ads_idcss_from_trusted_structure(initial_structure, ads_mol, bond_scale_factor=bond_scale_factor)
    #     broken = structure_is_broken_multi_checker(structure, initial_structure, ads_idcss, bond_scale_factor=bond_scale_factor, trust_initial_structure=trust_initial_structure)
    # return broken

def structure_is_broken_multi_checker(structure, initial_structure, ads_idcss: list[list[int]], bond_scale_factor=1.2, trust_initial_structure = False):
    # ads_idcss = get_ads_idcs(structure, ads_mol, safe_structure=initial_structure, bond_scale_factor=bond_scale_factor)
    for i, ads_idcs in enumerate(ads_idcss):
        other_ads_idcs = [idx for j, ads_idcs2 in enumerate(ads_idcss) if j != i for idx in ads_idcs2]
        for ads_idx in ads_idcs:
            other_internal_ads_idcs = [idx for idx in ads_idcs if idx != ads_idx]
            # if > 0 and not any(is_bonded(structure, ads_idx, idx, scale_factor=bond_scale_factor) for idx in other_internal_ads_idcs):
            # if trust_initial_structure:
            if True:
                # Check if this ads_idx has any bonds to other ads_idcs of the same ads_mol
                if len(other_internal_ads_idcs) and (not any(is_bonded(structure, ads_idx, idx, scale_factor=bond_scale_factor) for idx in other_internal_ads_idcs)):
                    print(f"structure of composition {structure.composition} is broken: ads_idx={ads_idx} ({structure[ads_idx]}) has no bonds to other ads atoms in the same ads_mol (within {other_internal_ads_idcs})")
                    return True
            # Check the other ads_mols to detect if unwanted bonds form between adsorbates
            other_mutual_ads_idcs = [idx for idx in ads_idcs if idx != ads_idx] + other_ads_idcs
            initially_bonded_idcs = [idx for idx in other_mutual_ads_idcs if is_bonded(initial_structure, ads_idx, idx, scale_factor=bond_scale_factor)]
            currently_bonded_idcs = [idx for idx in other_mutual_ads_idcs if is_bonded(structure, ads_idx, idx, scale_factor=bond_scale_factor)]
            if len(initially_bonded_idcs) == 0:
                continue
            else:
                if len(set(initially_bonded_idcs) - set(currently_bonded_idcs)) > 0:
                    print(f"structure of composition {structure.composition} is broken: ads_idx={ads_idx} ({structure[ads_idx]}) has lost bonds to other ads atoms")
                    print(f"  initially_bonded_idcs={initially_bonded_idcs}")
                    print(f"  currently_bonded_idcs={currently_bonded_idcs}")
                    print(f"  ads_idcs={ads_idcs}")
                    return True
    return False

def structure_is_broken(structure, initial_structure, ads_mol: str, bond_scale_factor=1.2, trust_initial_structure = True):
    ads_idcs = get_ads_idcs(structure, ads_mol)
    if surf_of_structure_is_broken(structure, ads_idcs, bond_scale_factor=bond_scale_factor):
        return True
    if not "_" in ads_mol:
        return structure_is_broken_singular(structure, initial_structure, ads_idcs, bond_scale_factor=bond_scale_factor)
    else:
        if trust_initial_structure:
            try:
                ads_idcs = get_ads_idcss_from_trusted_structure(initial_structure, ads_mol, bond_scale_factor=bond_scale_factor)
            except ValueError as e:
                print(f"Warning: could not get ads_idcs from trusted structure: {e}")
                print(f"  Falling back to default ordering ads_idcss")
                ads_idcs = get_ads_idcs(structure, ads_mol)
        return structure_is_broken_multi(structure, initial_structure, ads_idcs, ads_mol, bond_scale_factor=bond_scale_factor, trust_initial_structure=trust_initial_structure)


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
    log_generic(log_path1, log_path2, bond_scale_factor)

def _log_is_not_broken(calc_root: Path, bond_scale_factor: float):
    log_path1 = calc_root / "broken.txt"
    log_path2 = calc_root / "intact.txt"
    log_generic(log_path2, log_path1, bond_scale_factor)

def _clear_broken_log(calc_root: Path):
    log_path1 = calc_root / f"broken.txt"
    log_path2 = calc_root / f"intact.txt"
    for path in [log_path1, log_path2]:
        if path.exists():
            path.unlink()


def calc_root_log_shows_broken(calc_root: Path, bond_scale_factor: float) -> bool | None:
    if _calc_root_is_broken(calc_root, bond_scale_factor):
        return True
    if _calc_root_is_not_broken(calc_root, bond_scale_factor):
        return False
    return None

def calc_root_is_broken_operate(
        calc_root: Path, ads_mol: str, 
        bond_scale_factor=1.4, 
        write_log=True, 
        get_expected_path: Callable[[Path], Path] | None = None, 
        calc_root_is_finished: Callable[[Path], bool] | None = None
        ) -> bool | None:
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
        else:
            _clear_broken_log(calc_root)
    return is_broken

exception_mols = [
    "O", "H", "N",
]


def calc_root_is_broken(
        calc_root: Path, ads_mol: str, 
        bond_scale_factor=1.4, 
        check_log=True, write_log=True, 
        test_new_algo: bool = False, 
        get_expected_path: Callable[[Path], Path] | None = None, 
        calc_root_is_finished: Callable[[Path], bool] | None = None
        ) -> bool | None:
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

