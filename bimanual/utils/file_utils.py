import os

# ============================================================
# ENSURE DIRECTORY EXISTS
# ============================================================

def ensure_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )

# ============================================================
# GET FILE NAME WITHOUT EXTENSION
# ============================================================

def get_stem(path):

    filename = os.path.basename(path)

    stem = os.path.splitext(
        filename
    )[0]

    return stem

# ============================================================
# GET FILE EXTENSION
# ============================================================

def get_extension(path):

    return os.path.splitext(path)[1]

# ============================================================
# IS IMAGE FILE
# ============================================================

def is_image_file(filename):

    valid_exts = [

        ".png",
        ".jpg",
        ".jpeg",
        ".bmp"
    ]

    ext = get_extension(
        filename
    ).lower()

    return ext in valid_exts

# ============================================================
# SORT NATURALLY
# ============================================================

def natural_sort(strings):

    return sorted(strings)

# ============================================================
# PRINT PATH
# ============================================================

def print_path_info(path):

    print(f"\nPath: {path}")