import glob
import PIL.Image


def main():
    l = glob.glob('D:/students/dnn/assignment2/training/labels/*')

    lsize = len(l)
    pixels_values = set()
    widther, heighter = 0, 0
    for i, f in enumerate(l):
        if f[-4:] != '.png':
            print('Skipping file: {} '.format(f))
            continue
        print('{} / {} '.format(i, lsize), end='\r')
        im = PIL.Image.open(f).convert('RGB')  # note: should be np.array(im)
        colors = list(map(lambda x: x[1], im.getcolors()))
        for c in colors:
            pixels_values.add(c)
        size = im.size
        if size[0] > size[1]:
            widther += 1
        else:
            heighter += 1

    print('\nLarger: width {}, height: {}'.format(widther, heighter))
    print('Values:\n\tmin: {},\n\tmax: {},\n\tdifferent values: {}'.format(
          None, None,  # min(pixels_values), max(pixels_values),
          len(pixels_values)))
    print('Values:')
    print(pixels_values)


if __name__ == '__main__':
    main()

# Out:
# (tf) aj370953@sylvester:~$ srun /home/aj370953/.virtualenvs/tf/bin/python3.5 test.py
# Skipping file: /scidata/assignment2/training/labels/sample2.zip
# 18000 / 18001
# Values:
#         min: 0,
#         max: 65,
#         different values: 66
# Values:
# {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65}
# (tf) aj370953@sylvester:~$
