import logging
import os
from pathlib import Path
from typing import Any, NewType, cast

FilesystemStructure = NewType("FilesystemStructure", dict[str, Any])


def create_structure(root_path: Path, structure: FilesystemStructure) -> None:
    """
    Create files and directories according to the structure.
    """

    def create_structure_internal(
        root_path: Path, cur_path: Path, structure: FilesystemStructure
    ) -> None:
        for path, content in structure.items():
            full_path: Path = cur_path / path

            if isinstance(content, dict):  # Directory
                full_path.mkdir(parents=True, exist_ok=True)
                create_structure_internal(
                    root_path,
                    full_path,
                    FilesystemStructure(cast(dict[str, Any], content)),
                )
            elif isinstance(content, tuple) and content[0] == "file":
                assert len(content) == 1
                full_path.touch()
            elif isinstance(content, tuple) and content[0] == "symlink":
                assert len(content) == 2
                target_path = root_path / content[1]
                os.symlink(target_path, full_path)
            else:
                raise ValueError(f"Unsupported type for path ({path}): {content}")

    root_path.mkdir(parents=True, exist_ok=True)
    create_structure_internal(root_path, root_path, structure)


def verify_structure(root_path: Path, structure: FilesystemStructure) -> bool:
    """
    Verify that the files and directories match the expected structure.
    """

    def verify_structure_internal(
        root_path: Path, cur_path: Path, structure: FilesystemStructure
    ) -> bool:
        # Check for the presence of each item specified in the structure
        for name, item in structure.items():
            new_cur_path = cur_path / name
            if not path_exists_dont_follow_symlinks(new_cur_path):
                logging.debug(f"{new_cur_path} does not exist")
                return False
            elif isinstance(item, dict):
                if not new_cur_path.is_dir():
                    logging.debug(f"expected {new_cur_path} to be a directory")
                    return False
                if not verify_structure_internal(
                    root_path,
                    new_cur_path,
                    FilesystemStructure(cast(dict[str, Any], item)),
                ):
                    return False
            elif isinstance(item, tuple) and item[0] == "file":
                if not new_cur_path.is_file():
                    logging.debug(f"expected {new_cur_path} to be a regular file")
                    return False
            elif isinstance(item, tuple) and item[0] == "symlink":
                if not new_cur_path.is_symlink():
                    logging.debug(f"expected {new_cur_path} to be a symlink")
                    return False
                # If item[1] is None, this indicates that we expect the symlink to be broken
                if item[1] != None:
                    expected_target = root_path / item[1]
                    if not new_cur_path.resolve().samefile(expected_target):
                        logging.debug(
                            f"expected {new_cur_path} to link to {expected_target}, but it links to {new_cur_path.resolve()}"
                        )
                        return False
            else:
                assert False, "structure misconfigured"

        # Check for any extra files or directories not described by the structure
        expected_names = set(structure.keys())
        actual_names = {entry.name for entry in cur_path.iterdir()}
        if not expected_names.issuperset(actual_names):
            logging.debug(
                f"expected_names={expected_names}, actual_names={actual_names}"
            )
            return False

        return True

    if not root_path.exists():
        logging.debug(f"{root_path} does not exist")
        return False
    return verify_structure_internal(root_path, root_path, structure)


def make_workspace_structure(
    symlinks_structure: FilesystemStructure,
    task_runs_structure: FilesystemStructure,
) -> FilesystemStructure:
    """
    This function exists so that it's easier to refactor the tests in case we ever change
        how the workspace is organized.
    """
    return FilesystemStructure(
        {
            "dbgym_workspace": {
                "symlinks": symlinks_structure,
                "task_runs": task_runs_structure,
                "tmp": {},
            }
        }
    )


def path_exists_dont_follow_symlinks(path: Path) -> bool:
    """
    As of writing this comment, ray is currently constraining us to python <3.12. However, the "follow_symlinks" option in
    Path.exists() only comes up in python 3.12. Thus, this is the only way to check if a path exists without following symlinks.
    """
    # If the path exists and is a symlink, os.path.islink() will be true (even if the symlink is broken)
    if os.path.islink(path):
        return True
    # Otherwise, we know it's either non-existent or not a symlink, so path.exists() works fine
    else:
        return path.exists()
