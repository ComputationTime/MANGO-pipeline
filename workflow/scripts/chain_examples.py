"""Optional per-record target selection after antigen/antibody group splitting."""


def example_chains(row):
    value = row.get('example_target_chains') if isinstance(row, dict) else getattr(row, 'example_target_chains', None)
    if value is None:
        return ('heavy', 'light')
    chains = tuple(value.split(';'))
    if not chains or len(chains) != len(set(chains)) or not set(chains) <= {'heavy', 'light'}:
        raise ValueError(f'Invalid example_target_chains: {value!r}')
    return chains
