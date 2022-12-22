import argparse
from src.train import train
from src.test import test

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', dest='model', help='Model name', required=True)
    arg('--rank', dest='rank', help='Rank list', required=True)
    arg('--scheme', dest='scheme', help='Scheme', required=True)
    arg('--mode', dest='mode', help='Tarin or test mode', required=True)

    parser.set_defaults(model='CaffeBNAlexNet')
    parser.set_defaults(weights=1)
    parser.set_defaults(scheme='scheme_2')
    args = parser.parse_args()
    print(args.mode)
    if args.mode == 'train':
        time = train(args.model, args.rank, args.scheme)
        print(time)
    else:
        test(args.model, args.rank, args.scheme)




if __name__ == '__main__':
    main()
