"""Run one job through the installed SCALE-Sim Python API with traces disabled."""
import argparse
import inspect
from pathlib import Path
import sys

# Support both an installed package and the assignment repository checkout.
sys.path.insert(0, str(Path.cwd()))
from scalesim.scale_sim import scalesim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layout", help="Layout CSV if required by your assignment setup")
    args = parser.parse_args()
    kwargs = dict(save_disk_space=True, verbose=True, config=args.config,
                  topology=args.topology, input_type_gemm=True)
    layout = inspect.signature(scalesim).parameters.get("layout")
    if args.layout is not None:
        kwargs["layout"] = args.layout
    elif layout is not None and layout.default is inspect.Parameter.empty:
        parser.error("This install requires a layout. Add --layout and its absolute path "
                     "to the command list in settings.json.")
    scalesim(**kwargs).run_scale(top_path=args.output)


if __name__ == "__main__":
    main()
