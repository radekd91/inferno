import os
import sys
import shutil

def symlink_cp(src, dst):
    if os.path.isdir(src):
        if not os.path.exists(dst):
            os.makedirs(dst)
        files = os.listdir(src)
        for file in files:
            src_file = os.path.join(src, file)
            dst_file = os.path.join(dst, file)
            if os.path.isdir(src_file):
                symlink_cp(src_file, dst_file)
            else:
                os.symlink(src_file, dst_file)
    else:
        os.symlink(src, dst)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cp_symlink.py <source_dir> <destination_dir>")
        sys.exit(1)

    source_dir = sys.argv[1]
    destination_dir = sys.argv[2]
    
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist")
        sys.exit(1)

    symlink_cp(source_dir, destination_dir)
