"""An increased antigen cap must preserve the frozen small-cohort targets."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts'))
from prepare_asd_prototype import diverse_subset
from prepare_asd_scaling_8k import assert_nested
from prepare_asd_transfer import select_examples


class ASD8kTests(unittest.TestCase):
    def test_larger_cap_preserves_prefix_even_with_chain_duplicates(self):
        rows = [dict(id=f'{ag}_{i}', antigen_id=ag, resolved_ag_seq=ag,
                     resolved_H_seq=f'H{i // 2}', resolved_L_seq=f'L{i % 4}', split='train')
                for ag in ('AAA', 'CCC', 'DDD') for i in range(20)]
        small = select_examples(diverse_subset(rows, len(rows), 3, 0))
        large = select_examples(diverse_subset(rows, len(rows), 20, 0))
        assert_nested(small, large)
        self.assertGreater(len(large), len(small))
        self.assertGreater(max(Counter(r['antigen_id'] for r in large).values()), 3)

    def test_same_record_ids_with_changed_chain_masks_are_rejected(self):
        small = [dict(id='record', example_target_chains='heavy;light')]
        large = [dict(id='record', example_target_chains='heavy')]
        with self.assertRaises(ValueError):
            assert_nested(small, large)


if __name__ == '__main__':
    unittest.main()
