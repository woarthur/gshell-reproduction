"""Train the three GShell stages using posed images."""

from argparse import ArgumentParser

from gshell.config import GShellConfig


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source", required=True, help="COLMAP or Blender dataset directory")
    parser.add_argument("-m", "--model-path", required=True, help="Output directory")
    parser.add_argument("--config", help="JSON configuration; omitted keys use paper/reference defaults")
    parser.add_argument("--resume", help="GShell checkpoint; stored settings override --config")
    parser.add_argument("--device", help="Override cuda/cpu device")
    args = parser.parse_args()
    config = GShellConfig.load(args.config)
    if args.device:
        config.device = args.device
    config.validate()
    from gshell.training import train
    train(args.source, args.model_path, config, args.resume)


if __name__ == "__main__":
    main()
