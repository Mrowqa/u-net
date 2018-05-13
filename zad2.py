import argparse
import train_valid_split


def main():
    parser = argparse.ArgumentParser(prog="zad2")
    subparsers = parser.add_subparsers(help="sub-command help")

    tvs = subparsers.add_parser('gen-splits', help="Generates splits from data")
    tvs.add_argument('data-dir', type=str, help="Data directory")
    tvs.add_argument('save-to', type=str, help="JSON file where split will be saved")
    tvs.add_argument('-s', '--splits', type=int, default=1, help="How many times split training data")
    tvs.add_argument('-v', '--small-valid', type=float, default=0.0,
                     help="What part of data destined for small validation set")

    # tf boards, etc?
    train = subparsers.add_parser('train', help="Trains a model")
    # model
    # split-config
    # ?batch-size
    # ?epochs
    # ?save-to

    eval = subparsers.add_parser('eval', help="Evaluates a model")
    # model
    # data-dir
    # ?batch-size
    # ?output-dir

    args = parser.parse_args()
    if hasattr(args, 'data-dir'):
        gen_train_valid_split(args)
    else:
        print(parser.format_help())


def gen_train_valid_split(args):
    dataset = train_valid_split.create_split_from_directory(getattr(args, 'data-dir'), args.splits, args.small_valid)
    train_valid_split.dump_to_file(dataset, getattr(args, 'save-to'))
    print('Done. Processed {} records.'.format(dataset["cnt"]))


if __name__ == '__main__':
    main()
