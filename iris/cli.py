import argparse
import ast
import iris.io_utils as io
from contextlib import suppress


def parse_overrides(expressions):
    overrides = {}
    for expr in expressions:
        key, value = expr.split('=')
        with suppress(SyntaxError):
            value = ast.literal_eval(value)
        overrides[key] = value
    return overrides


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--name', '-n', metavar='NAME', default=None, type=str, help='model name')
    parser.add_argument('--params', '-p', metavar='PARAMS', default=[], nargs="*", help='parameters overrides')


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train and evaluate TensorFlow model.')
    subparsers = parser.add_subparsers(title='action', dest="action")
    subparsers.required = True
    action_parsers = [subparsers.add_parser(action) for action in ['train', 'test']]
    for action_parser in action_parsers:
        add_common_arguments(action_parser)
    add_common_arguments(parser)

    return parser


class CLI:
    def __init__(self):
        parser = create_parser()
        args = parser.parse_args()

        self.action = args.action
        self.model_dir = io.get_model_dir(args.name)
        self.overrides = parse_overrides(args.params)
