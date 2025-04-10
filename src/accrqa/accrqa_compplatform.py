# See the LICENSE file at the top-level directory of this distribution.

import ctypes

class accrqaCompPlatform:
    """
    A class representing a coputational platform supported by AccRQA.
    
    This class maps supported computational platforms (e.g., "cpu", "nv_gpu")
    to integer identifiers.
    
    Args:
        comp_platform (str): The name of the computational platform to use. Must be one of supported computational platforms.
    
    Raises:
        ValueError: If the provided comp_platform is not supported.
    
    Supported computational platforms:
        cpu: CPU parallelised using OpenMP.
        
        nv_gpu: NVIDIA GPU parallelised using CUDA.
    """
    comp_platforms = {
        1: "cpu",
        1024: "nv_gpu"
    }

    def __init__(self, comp_platform: str):
        """
        Initialize an accrqaCompPlatform instance with a specified computational platform.
        
        Args:
            comp_platform (str): The name of the computational platform to use.
        
        Raises:
            ValueError: If the provided comp_platform is not recognized.
        """
        
        reverse_map = {v: k for k, v in self.comp_platforms.items()}
        if comp_platform not in reverse_map:
            raise ValueError(f"Unknown computational platform: {comp_platform}")

        self._distance = ctypes.c_int(reverse_map[comp_platform])

    def get_platform_id(self) -> int:
        """
        Get the integer ID associated with the selected computational platform.
        
        Returns:
            int: The internal computational platform ID.
        """
        return self._distance.value

    def get_platform_name(self) -> str:
        """
        Get the name of the computational platform associated with the internal ID.
        
        Returns:
            str: The name of the computational platform, or "Unknown" if not found.
        """
        return self.distance_types.get(self._distance.value, "Unknown platform")
