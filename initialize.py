import os
import sys

output_path = str(sys.argv[1])
isExist = os.path.exists(output_path)
if (not isExist):
    os.makedirs(output_path)
