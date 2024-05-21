# Author Loïc Thiriet

import re


def get_elements_with_regex(regex: str, str_list: list, unique: bool = True, authorize_no_match: bool = False):
    prog = re.compile(regex)
    matched_list = [prog.match(elt) for elt in str_list]
    filtered_matched_list = [elt.string for elt in matched_list if elt is not None]

    # Check if there is no matching pattern
    if len(filtered_matched_list) == 0:
        if not authorize_no_match:
            raise ValueError(f"No elments found matching the following regex='{regex}'.")
        else:
            return None

    # Check if there is one unique matching pattern
    if unique:
        if len(filtered_matched_list) == 1:
            return filtered_matched_list[0]
        else:
            raise ValueError(
                f"Found {len(filtered_matched_list)} elments matching the following regex='{regex}': "
                + f"{filtered_matched_list}"
            )
    else:
        return filtered_matched_list
