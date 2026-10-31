from __future__ import annotations

import argparse
import json

from grounding.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Visible GUI-grounding development tools")
    parser.add_argument("--config", default="models.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("server-commands")
    prepare = commands.add_parser("prepare-visible-sample")
    prepare.add_argument("--output", default="data/screenspot-sample")
    prepare.add_argument("--count", type=int, default=3)
    full = commands.add_parser("prepare-visible")
    full.add_argument("--output", default="data/screenspot-pro")
    full.add_argument("--limit", type=int)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("manifest")
    evaluate.add_argument("--candidate", default="submission.py")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--max-calls", type=int)
    evaluate.add_argument("--task", default="task.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "server-commands":
        import shlex

        from grounding.servers import server_command

        for model in config["models"]:
            print(shlex.join(server_command(config, model)))
    elif args.command == "prepare-visible-sample":
        from grounding.screenspot import prepare_visible_sample

        print(prepare_visible_sample(args.output, args.count))
    elif args.command == "prepare-visible":
        from grounding.screenspot import prepare_visible

        print(prepare_visible(args.output, args.limit))
    else:
        from grounding.evaluation import evaluate

        print(
            json.dumps(
                evaluate(
                    args.manifest,
                    args.candidate,
                    config,
                    args.output,
                    max_calls=args.max_calls,
                    task_path=args.task,
                    limit=args.limit,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
