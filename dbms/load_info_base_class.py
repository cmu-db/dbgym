from pathlib import Path
from typing import Optional


class LoadInfoBaseClass:
    """
    A base class for providing info for DBMSs to load the data of a benchmark
    When copying these functions to a specific benchmark's load_info.py file, don't
      copy the comments or type annotations or else they might become out of sync.
    """

    def get_schema_fpath(self) -> Path:
        raise NotImplemented

    def get_tables_and_fpaths(self) -> list[tuple[str, Path]]:
        raise NotImplemented

    # If the subclassing benchmark does not have constraints, you can return None here
    def get_constraints_fpath(self) -> Optional[Path]:
        raise NotImplemented
