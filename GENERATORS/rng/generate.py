from random import randrange
from alive_progress import alive_bar
import os

def count_files(dir):
    return len([1 for x in list(os.scandir(dir)) if x.is_file()])

# number lenght: 128
# target dir: ../../DATA/rng/
target_count = input()
script_dir = os.path.dirname(os.path.abspath(__file__));
target_file = script_dir + "/../../DATA/rng/" + str(count_files(script_dir + "/../../DATA/rng")) + ".txt"

def random_number(digits=128):
    n = 0;
    for i in range(digits):
        n = (n * 10) + randrange(10);
    return n;
    
def gen_rnumbers(count):
    nums = [];
    with open(target_file, "w") as f:
        for i in range(count):
            f.write(str(random_number()))
            if i != count - 1:  # only add newline if not last
                f.write("\n")
    
gen_rnumbers(int(target_count));