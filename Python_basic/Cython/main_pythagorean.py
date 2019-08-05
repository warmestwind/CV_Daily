# https://code.tutsplus.com/tutorials/speeding-python-with-cython--cms-29557
import time
# method 1
import pyximport;pyximport.install()
import pythagorean_triples
# method 2
import pythagorean_triples_naive
# method 3
# setup.py
# from distutils.core import setup
# from Cython.Build import cythonize
#
# setup(
#     ext_modules=cythonize("pythagorean_triples.pyx")
# )
# import pythagorean_triples

def main():
    start = time.time()
    result = pythagorean_triples.count(1000)
    duration = time.time() - start
    print(result, duration)

    import timeit
    print(timeit.timeit('count(1000)', setup='from pythagorean_triples import count', number=10))


if __name__ == '__main__':
    main()
