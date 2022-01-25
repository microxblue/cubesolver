import os
import sys
import time
import argparse
import usb.core
import usb.util
import cv2
import math
import random
import numpy as np
import kociemba

class STM32_USB_DEV:

    MAX_PKT      = 64
    CMD_PKT_LEN  = 12

    MY_VID  = 0x0686
    MY_PID  = 0x1023

    def __init__(self, devaddr, product, interface = 0):
        # find our device
        self.dev    = None
        self.infidx = None
        self.epout  = None
        self.epin   = None

        devs = usb.core.find(idVendor=STM32_USB_DEV.MY_VID,
                             idProduct=product,
                             find_all=True)

        if interface:
            infnum = 1
        else:
            infnum = 0

        tgts = []
        for dev in devs:
            cfg = next(iter(dev), None)
            infidx = 0
            for inf in iter(cfg):
                if inf.bInterfaceNumber == infnum:
                    tgts.append((dev, infidx))
                infidx = infidx + 1

        if len(tgts) > 0:
            if devaddr == '?':
                # list all devices
                for idx, tgt in enumerate(tgts):
                    print ('Device %d (%04X:%04X) at address %d' % (idx, tgt[0].idVendor, tgt[0].idProduct, tgt[0].address))
                sys.exit (-1)
            elif devaddr == '':
                self.dev, self.infidx = tgts[0]
            else:
                if devaddr.startswith('0x'):
                    addr = int(devaddr, 16)
                else:
                    addr = int(devaddr)
                for dev, infidx in tgts:
                    if dev.address == addr:
                        self.dev = dev
                        self.infidx = infidx

        if self.dev is None:
            raise SystemExit  ('Cannot find matched StmUsb device!')

        # set the active configuration. With no arguments, the first
        # configuration will be the active one
        if sys.platform == "win32":
            self.dev.set_configuration()

        # get an endpoint instance
        self.cfg  = self.dev.get_active_configuration()
        self.intf = self.cfg[(self.infidx, 0)]

        self.epout = usb.util.find_descriptor(
            self.intf,
            # match the first OUT endpoint
            custom_match=
            lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)

        self.epin = usb.util.find_descriptor(
            self.intf,
            # match the first OUT endpoint
            custom_match=
            lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

        if self.epout is None or self.epin is None:
            raise SystemExit ('Cannot find End Point!')



    def read (self, length = MAX_PKT, timeout = 100):
        try:
            data = self.dev.read(self.epin, length, timeout)
        except usb.USBError as e:
            err_str = repr(e)
            if ('timeout error' in err_str) or ('timed out' in err_str):
                data = b''
            else:
                data = None
        return data

    def write (self, data, timeout = 100):
        try:
            ret = self.dev.write(self.epout, data)
        except usb.USBError as e:
            if 'timeout error' in repr(e):
                ret = 0
            else:
                raise SystemExit ('\n%s' % repr(e))
        return len(data)

    def send_cmd (self, cmd):
        if len(cmd) <= STM32_USB_DEV.CMD_PKT_LEN:
            data = cmd + b'\x00' * (STM32_USB_DEV.CMD_PKT_LEN - len(cmd))
            self.write (data)

    def send_dat (self, dat):
        if len(dat) <= STM32_USB_DEV.MAX_PKT:
            data = dat + b'\x00' * (STM32_USB_DEV.MAX_PKT - len(dat))
            self.write (data)

    def close (self):
        usb.util.dispose_resources(self.dev)
        self.dev.reset()


def get_dominant_color(roi):
    average = roi.mean(axis=0).mean(axis=0)
    pixels = np.float32(roi.reshape(-1, 3))

    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return tuple(dominant.astype(np.uint8))


def color_sample (bgrcap, cam_box):
    x,y,w,h = cam_box
    s = 4
    clrs = []
    for j in range (3):
        for i in range (3):
            x1, y1 = (int(x + w*i/3 + w/6), int(y + h*j/3 + h/6))
            roi = bgrcap[y1-4:y1+4, x1-4:x1+4]
            ret = get_dominant_color (roi)
            clrs.append (ret)
    if len(clrs) == 9:
        return (clrs)
    else:
        return []


def color_diff (color_set, color_tst):
    # calculate distance between test color and the full set
    gap = []
    for each in color_set:
        gap.append (np.linalg.norm(each - color_tst))
    return gap


def color_classify (color_set, threshold):
    # flag
    color_res = [0] * color_set.shape[0]

    # group colors using the distance
    clr_grp = 1
    while True:
        if 0 not in color_res:
            break
        else:
            idx = color_res.index(0)
        gaps = color_diff (color_set, color_set[idx])
        for idx, gap in enumerate(gaps):
            if gap < threshold and color_res[idx] == 0:
                color_res[idx] = clr_grp
        clr_grp += 1

    return color_res


def color_detect (faces, threshold = 0):
    # flat colors
    color_set = np.array(faces).reshape(-1, 3)

    if threshold == 0:
        target_threshold = 0
        # auto check threshold
        for threshold in range (10, 200, 5):
            color_res = color_classify (color_set, threshold)
            if target_threshold == 0 and max(color_res) == 6:
                target_threshold = threshold + 10
                break
    else:
        target_threshold = threshold

    color_res = color_classify (color_set, target_threshold)
    if max(color_res) != 6 or len(set(color_res)) != 6:
        for i in range(0, len(color_res), 9):
            print (color_res[i:i+9])
        raise Exception ('Expect 6 different kind of colors in results !')

    # Fill in known colors
    color_map = {}
    for i in range(7):
        color_map[i] = chr(ord('1') + i)

    color_ord = 'grbo'
    for i in range(4):
        if i == 0:
            color_center = color_res[4+9*i]
        else:
            if color_center == color_res[4+9*i]:
                raise Exception ('Expect different center colors !')
        color_map[color_res[4+9*i]] = color_ord[i]
    color_all = ''.join([color_map[i] for i in color_res])

    # Determine remaining w and y
    color_tbd = {}
    for idx, each in enumerate(color_all):
        if each.isdigit():
            color_tbd[each] = color_set[idx]
    color_tbd_list = list(color_tbd)
    if len(color_tbd_list) != 2:
        print (color_all)
        raise Exception ('Expect 2 colors for unknow color!')

    color_tbd_id1  = color_tbd_list[0]
    rgb_var1 = np.var(np.asarray(color_tbd[color_tbd_list[0]]))
    rgb_var2 = np.var(np.asarray(color_tbd[color_tbd_list[1]]))

    # white color has less variation among R,G,B
    if rgb_var1 < rgb_var2:
        idx = 0
    else:
        idx = 1
    color_all = color_all.replace(color_tbd_list[idx], 'w').replace(color_tbd_list[1-idx], 'y')

    return color_all


def print_state (state):
    for i in range(len(state) // 9):
        i = i * 9
        print ('--------------')
        for j in range(3):
            print (state[i:i+3])
            i = i + 3

    print ('=====================')
    for i in range(len(state) // 9):
        i = i * 9
        print (state[i:i+9])


def wait_cam (cap, timeout):
    ts = time.time()
    while time.time() - ts < timeout:
        _, bgrcap = cap.read()
        time.sleep (.001)
    bgrcap = cv2.resize(bgrcap, (320, 240), interpolation = cv2.INTER_AREA)
    return bgrcap


def faces_to_notation (input):

    notation_dict = dict(zip('oybgwr', 'URFBLD'))

    cube_str = []

    # U
    cube_str.append (input[3][::-1])

    # R
    rline = input[4][6] + input[7][7] + input[5][8] + input[4][7] +'y' + input[5][7] + input[4][8] + input[8][7] + input[5][6]
    cube_str.append (rline)

    # F
    cube_str.append (input[2][::-1])

    # B
    cube_str.append (input[1][::-1])

    # L
    lline = input[6][2] + input[8][1] + input[5][0] + input[6][1] +'w' + input[5][1] + input[6][0] + input[9][1] + input[5][2]

    cube_str.append (lline)

    # D
    cube_str.append (input[0])

    notations = [notation_dict[i] for i in ''.join(cube_str)]

    notation_str = ''.join(notations)

    return notation_str


def detect_cube (file = ''):

    STEP_CAM_DELAY = 0.5

    stm_usb = STM32_USB_DEV('', 0x1023, 1)

    if file:
        image  = cv2.imread (file)
        bgrcap = image.copy ()
    else:
        """Find the cube in the webcam picture and grab the colors of the facelets."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        _, bgrcap = cap.read()

        if bgrcap is None:
            print('Cannot connect to webcam!')
            return

    height, width = bgrcap.shape[:2]

    CAM_BOX  =  (92, 57, 120, 120)
    cam_box  =  CAM_BOX

    clr      = ''
    state    = 0
    box_mode = 0
    cube_str = ''

    while True:

        # Take each frame
        if not file:
            _, bgrcap = cap.read()
        else:
            bgrcap = image.copy ()
        bgrcap = cv2.resize(bgrcap, (320, 240), interpolation = cv2.INTER_AREA)

        k = cv2.waitKey(5) & 0xFF
        if state != 0:
            k = 0

        if k == ord('c'):
            # Save current image to file
            cv2.imwrite ('cube.jpg', bgrcap)

        elif k == ord('x'):
            # Type x to exit
            break

        elif k == ord('m'):
            # Try to solve the cube
            if cube_str == '':
                print ("Please run scanning first !")
            else:
                print ("Resolving ...")
                res = []
                try:
                    ret   = kociemba.solve(cube_str)
                    steps = ret.split(' ')
                    for step in steps:
                        if len(step) == 1:
                            res.append(step + '1')
                        elif step.endswith("'"):
                            res.append(step[0] + '3')
                        else:
                            res.append(step)
                except:
                    pass

                if len(res):
                    steps = res
                    print ('%d moves: [%s]' % (len(steps), ' '.join(steps)))
                    if len(steps) < 30:
                        cmd = '%s' % ''.join(steps)
                        stm_usb.send_dat (cmd.encode())
                    else:
                        print ('Too many steps!')
                else:
                    print ('Incorrect cube state, might have errors in the scanning !')

        elif k == ord('b'):
            # Use fixed cam box
            cam_box  = CAM_BOX
            box_mode = 0

        elif k == ord('a'):
            # Use auto cam box
            box_mode = 3
            state = 100

        elif k == ord('d'):
            # Sacn cube faces
            print ("Scanning cube faces ...")
            faces = []
            state = 1

        elif k == ord('o'):
            # Release stepper motor
            print ("Release steppers")
            stm_usb.send_cmd (b'@ME0')

        elif k == ord('p'):
            # Lock stepper motor
            print ("Lock steppers")
            stm_usb.send_cmd (b'@ME1')

        elif k == ord('t'):
            # Rotate cube once
            print ("Rotate")
            stm_usb.send_cmd (b'@MT1')

        elif k == ord('n'):
            # Scramble the cube for 30 steps
            steps = 30
            print ("Scrambling cube for %d steps ..." % steps)
            dirs  = ['R', 'L', 'F', 'T']
            nums  = ['1', '2', '3']
            seq   = []
            tcnt  = 0
            last  = ''
            for i in range (steps):
                curr = random.choice(dirs)
                while curr == last:
                    curr = random.choice(dirs)
                snum = random.choice(nums)
                s = curr + snum
                seq.append (s)
                if curr == 'T':
                    tcnt += int(snum)
                last = curr
            # adjust so that the up side center piece does not change
            adj = (4 - (tcnt & 3)) & 3
            seq.append('T%d' % adj)
            cmd = '%s' % ''.join(seq)
            stm_usb.send_dat (cmd.encode())
            print ("Done !")

        sample  = 0
        cmd_str = b''
        if state > 0:
            if state < 5:
                print ('Scanning %d' % state)
                sample  = 1
                cmd_str = b'@MT1'
                state += 1

            elif state < 5 + 12:
                steps = [
                    'M1', 'T1',       # R_COL1 @ BOT
                    'T1', 'M1', 'T1', # L_COL1 @ TOP   R_COL3 @ BOT
                    'T1', 'N1', 'T1', # L_COL3 @ TOP
                    'T1', 'N1', 'T1', 'T1',
                ]
                idx = state - 5
                if idx in [2,5,8]:
                    print ('Scanning %d' % state)
                    sample = 1
                if steps[idx]:
                    cmd_str = b'@M%s' % steps[idx].encode()
                state += 1

            elif state < 17 + 12:
                steps = [
                    'T1', 'M1', 'T1', # RT @ BOT
                    'T1', 'M1', 'T1', # RB @ BOT  LB @ TOP
                    'T1', 'N1', 'T1', # LT @ TOP
                    'T1', 'N1', 'T1',
                ]
                idx = state - 17
                if idx in [3,6,9]:
                    print ('Scanning %d' % state)
                    sample = 1
                if steps[idx]:
                    cmd_str = b'@M%s' % steps[idx].encode()
                state += 1

            elif state == 100:

                    gray =  cv2.cvtColor(bgrcap,  cv2.COLOR_BGR2GRAY)
                    gray_blurred = cv2.blur(gray, (3, 3))
                    detected_circles = cv2.HoughCircles(gray_blurred,
                           cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                           param2 = 30, minRadius = 18, maxRadius = 25)
                    # Draw circles that are detected.
                    if detected_circles is not None:
                        # Convert the circle parameters a, b and r to integers.
                        detected_circles = np.uint16(np.around(detected_circles))
                        if len(detected_circles[0, :]) == 2 :
                            for pt in detected_circles[0, :]:
                                a, b, r = pt[0], pt[1], pt[2]

                                # Draw the circumference of the circle.
                                cv2.circle(bgrcap, (a, b), r, (0, 255, 0), 2)

                                # Draw a small circle (of radius 1) to show the center.
                                cv2.circle(bgrcap, (a, b), 1, (0, 0, 255), 3)

                            pts = detected_circles[0, :]
                            dist = math.sqrt((pts[0][0]*1.0 -  pts[1][0]*1.0)**2 + (pts[1][1]*1.0 - pts[1][1]*1.0)**2)
                            x = int((pts[0][0] + pts[1][0]) // 2)+2
                            y = int((pts[0][1] + pts[1][1]) // 2)+2
                            m = int(dist / 4.2)
                            cam_box = (x-m, y-m, 2 * m,  2 * m)
                            print ("Adjusted CAM box: ", cam_box)
                            box_mode = 1
                            state    = 0

            else:
                state = 0
                if 0:
                    print ('')
                    for face in faces:
                        print (face)

                #print ("Faces: %d" % len(faces))
                #print (faces)
                face_scans = eval(str(faces))
                result = color_detect (face_scans)
                #print_state (result)
                face_res = [result[i*9:i*9+9] for i in range(len(result)//9)]
                #print (face_res)
                cube_str = faces_to_notation (face_res)
                print (cube_str)

        if len(cmd_str) > 0:
            if sample:
                face_clrs = color_sample (bgrcap, cam_box)
                faces.append (face_clrs)
            stm_usb.send_cmd (cmd_str)
            bgrcap = wait_cam (cap, STEP_CAM_DELAY)

        x,y,w,h = cam_box
        if box_mode < 3:
            if box_mode == 0:
                c = (255,0,0)
            else:
                c = (255,255,0)
            cv2.rectangle(bgrcap, (x,y), (x+w, y+h), c, 2)

        cv2.imshow('Webcam - type "x" to quit.', bgrcap)

detect_cube ()
