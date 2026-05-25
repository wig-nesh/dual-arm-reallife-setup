import numpy as np

# ArUco marker configuration (3x4 board with 12 markers)
# Layout:
#  0  1  2
#  3  4  5
#  6  7  8
#  9 10 11
MARKER_LENGTH = 0.056  # 5.6cm markers
BOARD_WIDTH = 0.197  # edge-to-edge (3 markers wide)
BOARD_HEIGHT = 0.259  # edge-to-edge (4 markers tall)

# Convert edge-to-edge to center-to-center spacing
spacing_x = (BOARD_WIDTH - MARKER_LENGTH) / 2  # 3 markers = 2 intervals
spacing_y = (BOARD_HEIGHT - MARKER_LENGTH) / 3  # 4 markers = 3 intervals

# Build marker world positions (all 12 markers centered around origin)
half_extent_x = (BOARD_WIDTH - MARKER_LENGTH) / 2
half_extent_y = (BOARD_HEIGHT - MARKER_LENGTH) / 2

MARKER_WORLD = {}
for row in range(4):  # 4 rows
    for col in range(3):  # 3 columns
        marker_id = row * 3 + col  # 0-11
        x = -half_extent_x + col * spacing_x
        y = -half_extent_y + row * spacing_y
        MARKER_WORLD[marker_id] = np.array([x, y, 0.0])
