from id_unusable.id_broken import calc_root_is_broken
from id_unusable.id_desorbed import calc_root_is_desorbed
from id_unusable.helpers import get_outfile_path
from typing import Callable
from pathlib import Path


def calc_root_is_unusable(calc_root: Path, ads_mol: str, broken_bond_scale_factor=1.4, desorbed_bond_scale_factor=1.4, check_log=True, write_log=True, get_expected_path: Callable | None = None) -> bool | None:
    """
    Determine if a calculation root is unusable based on broken or desorbed adsorbates.
    
    Args:
        calc_root (Path): The path to the calculation root.
        ads_mol (str): The adsorbate molecule identifier.
        broken_bond_scale_factor (float): Scale factor for determining if bonds are broken (multiplied against sum of two atom types covalent radii to set maximum expected bond length).
        desorbed_bond_scale_factor (float): Scale factor for determining if adsorbates are desorbed.
        check_log (bool): Whether to check existing log files for previous results.
        write_log (bool): Whether to write log files for the current check.
        get_expected_path (Callable | None): Optional function to get the expected output file path from the calc_root as an argument, ie `lambda path: path / "opt" / "jdftx.out"`.
        """
    outfile_path = get_outfile_path(calc_root, get_expected_path=get_expected_path)
    if outfile_path is None:
        return None
    elif not outfile_path.exists():
        return True
    try:
        is_broken = calc_root_is_broken(calc_root, ads_mol, bond_scale_factor=broken_bond_scale_factor, check_log=check_log, write_log=write_log, get_expected_path=get_expected_path)
        if is_broken is None:
            return None
        elif is_broken:
            return True
        is_desorbed = calc_root_is_desorbed(calc_root, ads_mol, bond_scale_factor=desorbed_bond_scale_factor, check_log=check_log, write_log=write_log, get_expected_path=get_expected_path)
        if is_desorbed is None:
            return None
        elif is_desorbed:
            return True
        return False
    except Exception as e:
        return True