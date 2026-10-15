"""Export independent shell meshes, interior Gaussians and HDR illumination."""

from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from gshell.model import load_checkpoint
    from gshell.export import export_scene
    model, _ = load_checkpoint(args.checkpoint, args.device)
    if model.stage != 3:
        parser.error("Complete asset export needs a Stage III checkpoint")
    export_scene(model, args.output, args.resolution)


if __name__ == "__main__":
    main()
