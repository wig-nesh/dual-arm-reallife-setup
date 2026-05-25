import cv2

# Global state for SAM selection
state = {
    "mode": "box",  # default
    "points": [],
    "box": None,
    "drawing": False,
}


def mouse_callback(event, x, y, flags, param):
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        if state["mode"] == "point":
            state["points"] = [(x, y)]
        elif state["mode"] == "box":
            state["drawing"] = True
            state["box"] = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE:
        if state["drawing"] and state["mode"] == "box":
            state["box"][2] = x
            state["box"][3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        if state["drawing"] and state["mode"] == "box":
            state["drawing"] = False
            state["box"][2] = x
            state["box"][3] = y

