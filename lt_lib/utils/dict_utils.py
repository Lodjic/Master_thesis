# Author: Loïc Thiriet

from typing import Any


def add_nested_dict(dict1: dict, dict2: dict) -> dict:
    if isinstance(dict1, float) or isinstance(dict1, int):
        return dict1 + dict2
    else:
        for key in dict1.keys():
            dict1[key] = add_nested_dict(dict1[key], dict2[key])
        return dict1


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extend_flattened_dict(flattened_dict: dict[str, Any]):
    extended_dict = dict()
    for k, v in flattened_dict.items():
        subkeys = k.split(".")
        d = extended_dict
        for subkey in subkeys[:-1]:
            if subkey not in d:
                d[subkey] = dict()
            d = d[subkey]
        d[subkeys[-1]] = v
    return extended_dict


def keep_only_last_value_for_all_keys(d: dict) -> dict:
    items = []
    for k, v in d.items():
        items.append((k, v[-1]))
    return dict(items)


def append_nested_dict_with_0(nested_dict: dict) -> dict:
    for key in nested_dict.keys():
        if isinstance(nested_dict[key], dict):
            nested_dict[key] = append_nested_dict_with_0(nested_dict[key])
        elif isinstance(nested_dict[key], list):
            nested_dict[key].append(0)
        else:
            raise TypeError(f"metrics object should only contain dict and list not '{type(nested_dict[key])}'")
    return nested_dict
