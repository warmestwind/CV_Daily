#https://docs.python.org/3/library/argparse.html#module-argparse
import argparse
parser = argparse.ArgumentParser()
#positional arguments must be transmit
parser.add_argument("square", help="display a square of a given number",
                    type=int)
#optional arguments are unnecessary to transmit, dest ,default only in optional argu                 
parser.add_argument("--add", dest='bias',default=2, help="add a int to square", type=int

#if spycify minibatch then minibatch is True                              
parser.add_argument('--minibatch', action='store_true',
                     help='use minibatch discrimination')
# Parsing arguments
args = parser.parse_args()

#if args.bias:
print(args.square**2+args.bias)
#else :print(args.square**2) # py arg_parse.py 2 --add=1/--add 1
print(args.bias)

# usage:
# python arg_parse.py --minibatch                     
