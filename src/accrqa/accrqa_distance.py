# See the LICENSE file at the top-level directory of this distribution.

import ctypes

class accrqaDistance:
    """
    A class representing a distance type used in AccRQA.
    
    This class maps supported distance type names (e.g., "euclidean", "maximal")
    to integer identifiers.
    
    Args:
        distance_type (str): The name of the distance type to use. Must be one of supported distance types.
    
    Raises:
        ValueError: If the provided distance_type is not supported.
    
    Supported distance types:
        euclidian: L-2 norm or an euclidean norm.
        
        maximal: L-inf or a maximal norm.
    """
    distance_types = {
        1: "euclidean",
        2: "maximal"
    }

    def __init__(self, distance_type: str):
        """
        Initialize an accrqaDistance instance with a specified distance type.
        
        Args:
            distance_type (str): The name of the distance type to use.
        
        Raises:
            ValueError: If the provided distance_type is not recognized.
        """
        
        reverse_map = {v: k for k, v in self.distance_types.items()}
        if distance_type not in reverse_map:
            raise ValueError(f"Unknown distance type: {distance_type}")

        self._distance = ctypes.c_int(reverse_map[distance_type])

    def get_distance_id(self) -> int:
        """
        Get the integer ID associated with the selected distance type.
        
        Returns:
            int: The internal distance type ID.
        """
        return self._distance.value

    def get_distance_name(self) -> str:
        """
        Get the name of the distance type associated with the internal ID.
        
        Returns:
            str: The name of the distance type, or "Unknown" if not found.
        """
        return self.distance_types.get(self._distance.value, "Unknown distance")