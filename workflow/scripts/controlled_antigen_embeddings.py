"""Embedding-only entry point for the isolated ESM2 environment."""
import argparse
from pathlib import Path

from affinity_study_common import read_json
from asd_transfer_embeddings import embed_esm2


def main(root):
    root=Path(root);settings=read_json(root/'protocol.json')['settings']
    embed_esm2(dict(settings,experiment_root=str(root)),'embedding',str(root/'records.csv'),
               str(root/'embeddings_complete.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    main(**vars(parser.parse_args()))
