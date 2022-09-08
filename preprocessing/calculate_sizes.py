import os
from pathlib import Path
import math

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def main():
    pathlist = Path('../data').glob('**/*.pickle')
    open('data_sizes.txt', 'w').close()
    for path in pathlist:
        path_in_str = str(path)
        size = os.path.getsize(path_in_str)
        with open('data_sizes.txt', 'a+') as f:
            f.write(path_in_str)
            f.write('\n')
            f.write(convert_size(size))
            f.write('\n')
            f.close()

if __name__ == "__main__":
    main()