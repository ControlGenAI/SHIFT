"""Explicit commands only: importing this package never loads models or collects data."""
import argparse
from .runtime import read_config


def main():
    parser = argparse.ArgumentParser(description='SHIFT invertible DINO adapter: all double blocks, step 0')
    parser.add_argument('--config', default='configs/dino_adapter.json')
    sub = parser.add_subparsers(dest='command', required=True)
    collect = sub.add_parser('collect', help='Optional GPU-server collection; never run implicitly')
    collect.add_argument('--pairs', required=True)
    collect.add_argument('--output', required=True)
    collect.add_argument('--device', default='cuda:0')
    collect.add_argument('--resume', action='store_true',
                         help='Continue an interrupted collection from collection_progress.json')
    train = sub.add_parser('train', help='Train per-block adapters from saved dataset only')
    train.add_argument('--dataset', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--device', default='cuda:0')
    directions = sub.add_parser('directions', help='Train-pair directions; uses saved tensors, no models')
    directions.add_argument('--dataset', required=True)
    directions.add_argument('--output', required=True)
    steer = sub.add_parser('steer', help='Image post-block adapter/control sweeps, one block at a time')
    steer.add_argument('--dataset', required=True)
    steer.add_argument('--adapters', help='Required only for adapter_comparison mode')
    steer.add_argument('--directions', required=True)
    steer.add_argument('--output', required=True)
    steer.add_argument('--device', default='cuda:0')
    steer.add_argument('--split', choices=['val', 'test'], default='test')
    steer.add_argument('--resume', action='store_true', help='Reuse generations already on disk')
    evaluate = sub.add_parser('evaluate', help='Score saved outputs with DINO and pixel-change proxies')
    evaluate.add_argument('--directions', required=True)
    evaluate.add_argument('--results', required=True)
    evaluate.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = read_config(args.config)
    if args.command == 'collect':
        from .data import collect
        collect(config, args.pairs, args.output, args.device, args.resume)
    elif args.command == 'train':
        from .training import train
        train(config, args.dataset, args.output, args.device)
    elif args.command == 'directions':
        from .directions import build_directions
        build_directions(config, args.dataset, args.output)
    elif args.command == 'steer':
        from .steering import steer
        steer(config, args.dataset, args.adapters, args.directions, args.output, args.device,
              args.split, args.resume)
    elif args.command == 'evaluate':
        from .steering import evaluate
        evaluate(config, args.directions, args.results, args.device)


if __name__ == '__main__':
    main()
