import cv2
import mediapipe as mp
import mido
import numpy as np
from collections import defaultdict
import time

# At the top of the file, after imports:
FONT_FACE = cv2.FONT_HERSHEY_COMPLEX
FONT_COLOR = (255,255,255)
CURRENT_GESTURE_MSG = {'msg': ''}

# ——— MIDI assignments ———
TOGGLE_NOTE       = 60      # Play/Pause toggle (single button)
VOLUME_CC         = 7       # Mixer volume fader
TEMPO_CC          = 22      # Deck tempo slider
LOOP_NOTE         = 62      # Loop 1 beat toggle
LOOP2_NOTE        = 63      # Loop 2 beat toggle
BEAT_SYNC_NOTE    = 64      # Beat sync toggle
LOW_EQ_CC         = 14      # CC number for low EQ, adjust as needed
MID_EQ_CC         = 15      # CC number for mid EQ
HIGH_EQ_CC        = 16      # CC number for high EQ
STEM_DRUMS_NOTE   = 70
STEM_VOCALS_NOTE   = 71
STEM_INST_NOTE     = 72

# ——— Gesture thresholds ———
OPEN_THRESH       = 0.5     # threshold for open palm
CLOSE_THRESH      = 0.5     # threshold for closed fist
DEBOUNCE_FRAMES   = 2
DEBOUNCE_TIME     = 0
VOL_THRESHOLD     = 3
TEMPO_THRESHOLD   = 3
LOOP_DEBOUNCE_FRAMES = 3
LOOP_DEBOUNCE_TIME   = 0.05
BEAT_SYNC_DEBOUNCE_FRAMES = 1
BEAT_SYNC_DEBOUNCE_TIME = 0

# ——— Default-zone settings ———
# Default zones are small squares of size 25% of the smaller screen dimension
ZONE_SIZE_FACTOR  = 0.15    # fraction of min(width,height)
DEFAULT_CC        = 64      # middle CC value
DEFAULT_VOLUME_WIDTH = 0.1

# ——— Helper functions ———
def palm_distance(lm):
    wrist = np.array([lm[0].x, lm[0].y])
    tips  = np.array([[lm[i].x, lm[i].y] for i in (4,8,12,16,20)])
    return np.mean(np.linalg.norm(tips - wrist, axis=1))

# hand-state checks
def is_hand_open(lm):
    # Open if palm spread exceeds threshold
    return palm_distance(lm) > OPEN_THRESH

def is_hand_closed(lm, h):
    # Closed if palm spread below threshold and all fingers folded
    if palm_distance(lm) > CLOSE_THRESH:
        return False
    for tip,pip in ((8,6),(12,10),(16,14),(20,18)):
        if lm[tip].y * h < lm[pip].y * h:
            return False
    return True

# thumb up/down checks
def is_thumbs_up(lm, h):
    if lm[4].y * h >= lm[2].y * h:
        return False
    for tip,pip in ((8,6),(12,10),(16,14),(20,18)):
        if lm[tip].y * h < lm[pip].y * h:
            return False
    return True

def is_thumbs_down(lm, h):
    if lm[4].y * h <= lm[2].y * h:
        return False
    for tip,pip in ((8,6),(12,10),(16,14),(20,18)):
        if lm[tip].y * h < lm[pip].y * h:
            return False
    return True

def is_index_up(lm, h):
    # index tip above PIP and other fingers folded
    if lm[8].y * h >= lm[6].y * h:
        return False
    for tip,pip in ((12,10),(16,14),(20,18)):
        if lm[tip].y * h < lm[pip].y * h:
            return False
    return True

def is_two_fingers_up(lm, h):
    # index and middle tips above their PIPs
    if lm[8].y * h >= lm[6].y * h or lm[12].y * h >= lm[10].y * h:
        return False
    # other fingers folded
    for tip,pip in ((16,14),(20,18)):
        if lm[tip].y * h < lm[pip].y * h:
            return False
    return True

def palm_normal(lm):
    # Use wrist (0), index_mcp (5), pinky_mcp (17)
    p0 = np.array([lm[0].x, lm[0].y, lm[0].z])
    p1 = np.array([lm[5].x, lm[5].y, lm[5].z])
    p2 = np.array([lm[17].x, lm[17].y, lm[17].z])
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    return normal

def is_palm_facing_camera(lm, label=None):
    normal = palm_normal(lm)
    # For right hand, the normal direction is inverted
    if label == 'Right':
        return normal[2] > 0
    else:
        return normal[2] < 0

def count_fingers_up_back(lm, h):
    # For back of hand, finger up means tip is above PIP in y (screen coordinates)
    up = []
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        up.append(lm[tip].y * h < lm[pip].y * h)
    return up  # [index, middle, ring, pinky]

def is_finger_up(lm, tip, pip, h, min_dist=0.04):
    return (lm[tip].y * h < lm[pip].y * h) and (abs(lm[tip].y - lm[pip].y) > min_dist)

def is_index_and_middle_up(lm, h):
    # index and middle tips above their PIPs, others folded
    if lm[8].y * h >= lm[6].y * h or lm[12].y * h >= lm[10].y * h:
        return False
    for tip,pip in ((16,14),(20,18)):
        if lm[tip].y * h < lm[pip].y * h:
            return False
    return True

def is_middle_ring_pinky_up(lm, h):
    # middle, ring, pinky tips above their PIPs, index folded
    if lm[8].y * h < lm[6].y * h:
        return False
    for tip,pip in ((12,10),(16,14),(20,18)):
        if lm[tip].y * h >= lm[pip].y * h:
            return False
    return True

# Add a helper to update the gesture message
def set_gesture_msg(msg):
    CURRENT_GESTURE_MSG['msg'] = msg

def get_eq_zones(label, w, h):
    zone_size = int(min(w, h) * ZONE_SIZE_FACTOR)
    offset = 120
    if label == 'Left':
        tempo_bar_x = 250
        eq_x1 = tempo_bar_x + int(DEFAULT_VOLUME_WIDTH * w * 0.75) + 20
        eq_x2 = eq_x1 + zone_size
    else:
        tempo_bar_x = w - 250 - int(DEFAULT_VOLUME_WIDTH * w * 0.75)
        eq_x2 = tempo_bar_x - 20
        eq_x1 = eq_x2 - zone_size
    inc_y1 = int((h - 3 * zone_size - 40) / 2) + offset
    inc_y2 = inc_y1 + zone_size
    dec_y1 = inc_y2 + 20
    dec_y2 = dec_y1 + zone_size
    mode_y1 = dec_y2 + 20
    mode_y2 = mode_y1 + zone_size
    inc_zone = (eq_x1, inc_y1, eq_x2, inc_y2)
    dec_zone = (eq_x1, dec_y1, eq_x2, dec_y2)
    mode_zone = (eq_x1, mode_y1, eq_x2, mode_y2)
    return {'inc': inc_zone, 'dec': dec_zone, 'mode': mode_zone}

# ——— Handlers ———
def handle_playpause(label, gesture, trackers, outport, palm_forward, lm, frame):
    if lm and frame is not None:
        h, w, _ = frame.shape
        zones = get_eq_zones(label, w, h)
        mode_zone = zones['mode']
        x = lm[0].x * w
        y = lm[0].y * h
        in_mode = (mode_zone[0] <= x <= mode_zone[2] and mode_zone[1] <= y <= mode_zone[3])
        if in_mode:
            return
            
    # gesture is 'IndexUp' for index finger up only
    playing = trackers['is_playing']
    if gesture != 'IndexUp' or not palm_forward:
        trackers['last_gesture'][label] = None
        return
    # Toggle logic: only trigger on rising edge
    if trackers['last_gesture'][label] != gesture:
        trackers['last_gesture'][label] = gesture
        # toggle play/pause
        playing[label] = not playing[label]
        ch = 0 if label == 'Left' else 1
        outport.send(mido.Message('note_on', channel=ch,
                                  note=TOGGLE_NOTE, velocity=127))
        state = 'Play' if playing[label] else 'Pause'
        print(f"{label} hand IndexUp → {state} (ch{ch})")
        set_gesture_msg(f"{label} hand IndexUp → {state} (ch{ch})")


def handle_volume(label, open_flag, lm, vs, outport, frame):
    h, w, _ = frame.shape
    # Thinner bar
    bar_w = int(DEFAULT_VOLUME_WIDTH * w * 0.75)
    if label == 'Left':
        bar_x = 50
    else:
        bar_x = w - 50 - bar_w
    x = lm[0].x * w if open_flag else None
    bar_color = (0, 0, 0)  # Solid black
    cv2.rectangle(frame, (bar_x, 500), (bar_x+bar_w, h), bar_color, -1)
    # Draw current indicator for both bars
    for side, color in zip(['Left', 'Right'], [(0,255,0), (255,0,0)]):
        last = vs['last'][side]
        if last >= 0:
            ch = 0 if side == 'Left' else 1
            top = 500
            bottom = frame.shape[0]
            if side == 'Left':
                x1, x2 = 50, 50+bar_w
            else:
                x1, x2 = w-50-bar_w, w-50
            y_pos = int(bottom - (last / 127) * (bottom - top))
            cv2.rectangle(frame, (x1, y_pos-2), (x2, y_pos+2), (255,255,255,128), -1)
    # Draw GAIN text
    font = FONT_FACE
    font_scale = 1.0
    thickness = 1
    text = 'GAIN'
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    while text_size[0] > bar_w - 10 and font_scale > 0.3:
        font_scale -= 0.05
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = bar_x + (bar_w - text_size[0]) // 2
    text_y = 495
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, FONT_COLOR, thickness, cv2.LINE_AA)
    if not open_flag:
        return
    if label == 'Left' and (x<50 or x > bar_w+50):
        return
    if label == 'Right' and (x < w - bar_w-50 or x > w-50):
        return
    y = lm[0].y * h
    top=500
    bottom = frame.shape[0]
    last = vs['last'][label]
    indicator_y = int(bottom - (last / 127) * (bottom - top))
    touch_threshold = 50  # pixels
    if abs(y - indicator_y) > touch_threshold:
        return
    cc = int(np.clip((bottom - y) / (bottom - top) * 127, 0, 127))
    if abs(cc - last) >= vs['threshold']:
        vs['last'][label] = cc
        ch = 0 if label == 'Left' else 1
        outport.send(mido.Message('control_change', channel=ch,
                                  control=VOLUME_CC, value=cc))
        print(f"{label} Volume → {cc} (ch{ch})")
        set_gesture_msg(f"{label} Volume → {cc} (ch{ch})")

def handle_tempo(label, closed_flag, lm, ts, outport, frame):
    h, w, _ = frame.shape
    # Thinner bar
    bar_w = int(DEFAULT_VOLUME_WIDTH * w * 0.75)
    if label == 'Left':
        bar_x = 250
    else:
        bar_x = w - 250 - bar_w
    x = lm[0].x * w if closed_flag else None
    bar_color = (0, 0, 0)  # Solid black
    cv2.rectangle(frame, (bar_x, 500), (bar_x+bar_w, h), bar_color, -1)
    # Draw current indicator for both bars
    for side, color in zip(['Left', 'Right'], [(0,255,0), (255,0,0)]):
        last = ts['last'][side]
        if last >= 0:
            ch = 0 if side == 'Left' else 1
            top = 500
            bottom = frame.shape[0]
            if side == 'Left':
                x1, x2 = 250, 250+bar_w
            else:
                x1, x2 = w-250-bar_w, w-250
            y_pos = int(bottom - (last / 127) * (bottom - top))
            cv2.rectangle(frame, (x1, y_pos-2), (x2, y_pos+2), (255,255,255,128), -1)
    # Draw TEMPO text
    font = FONT_FACE
    font_scale = 1.0
    thickness = 1
    text = 'TEMPO'
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    while text_size[0] > bar_w - 10 and font_scale > 0.3:
        font_scale -= 0.05
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = bar_x + (bar_w - text_size[0]) // 2
    text_y = 495
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, FONT_COLOR, thickness, cv2.LINE_AA)
    if not closed_flag:
        return
    if label == 'Left' and (x<250 or x > bar_w+250):
        return
    if label == 'Right' and (x < w - bar_w-250 or x > w-250):
        return
    y = lm[0].y * h
    top=500
    bottom = frame.shape[0]
    last = ts['last'][label]
    indicator_y = int(bottom - (last / 127) * (bottom - top))
    touch_threshold = 50  # pixels
    if abs(y - indicator_y) > touch_threshold:
        return
    cc = int(np.clip((bottom - y) / (bottom - top) * 127, 0, 127))
    if abs(cc - last) >= ts['threshold']:
        ts['last'][label] = cc
        ch = 0 if label == 'Left' else 1
        outport.send(mido.Message('control_change', channel=ch,
                                  control=TEMPO_CC, value=cc))
        print(f"{label} Tempo → {cc} (ch{ch})")
        set_gesture_msg(f"{label} Tempo → {cc} (ch{ch})")

def handle_eq(label, open_flag, lm, outport, frame, eq_state, gestures, palm_forward):
    h, w, _ = frame.shape
    zones = get_eq_zones(label, w, h)
    inc_zone = zones['inc']
    dec_zone = zones['dec']
    mode_zone = zones['mode']
    zone_size = int(min(w, h) * ZONE_SIZE_FACTOR)
    
    # Draw UI
    inc_color = (0, 0, 0)
    dec_color = (0, 0, 0)
    mode_color = (0, 0, 0)
    cv2.rectangle(frame, (inc_zone[0], inc_zone[1]), (inc_zone[2], inc_zone[3]), inc_color, -1)
    cv2.rectangle(frame, (dec_zone[0], dec_zone[1]), (dec_zone[2], dec_zone[3]), dec_color, -1)
    cv2.rectangle(frame, (mode_zone[0], mode_zone[1]), (mode_zone[2], mode_zone[3]), mode_color, -1)
    
    # Draw text
    font = FONT_FACE
    thickness = 1
    for zone, text in [(inc_zone, 'INC. EQ'), (dec_zone, 'DEQ. EQ')]:
        font_scale = 0.9
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        while text_size[0] > zone_size - 10 and font_scale > 0.3:
            font_scale -= 0.05
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = zone[0] + (zone_size - text_size[0]) // 2
        text_y = zone[1] + (zone_size + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, FONT_COLOR, thickness, cv2.LINE_AA)
    mode_text = f"MODE: {eq_state['mode'][label].upper()}"
    font_scale = 0.7
    text_size = cv2.getTextSize(mode_text, font, font_scale, thickness)[0]
    while text_size[0] > zone_size - 10 and font_scale > 0.3:
        font_scale -= 0.05
        text_size = cv2.getTextSize(mode_text, font, font_scale, thickness)[0]
    text_x = mode_zone[0] + (zone_size - text_size[0]) // 2
    text_y = mode_zone[1] + (zone_size + text_size[1]) // 2
    cv2.putText(frame, mode_text, (text_x, text_y), font, font_scale, FONT_COLOR, thickness, cv2.LINE_AA)

    if not lm:
        return

    x = lm[0].x * w
    y = lm[0].y * h
    
    in_inc = (inc_zone[0] <= x <= inc_zone[2] and inc_zone[1] <= y <= inc_zone[3])
    in_dec = (dec_zone[0] <= x <= dec_zone[2] and dec_zone[1] <= y <= dec_zone[3])
    in_mode = (mode_zone[0] <= x <= mode_zone[2] and mode_zone[1] <= y <= mode_zone[3])
    
    if in_mode and palm_forward:
        active_gesture = None
        if gestures['two_beat_loop']:
            active_gesture = 'high'
        elif gestures['one_beat_loop']:
            active_gesture = 'mid'
        elif gestures['index_up']:
            active_gesture = 'low'
        
        if active_gesture and active_gesture != eq_state['last_gesture'][label]:
            eq_state['mode'][label] = active_gesture
            set_gesture_msg(f"{label} EQ Mode: {active_gesture.upper()}")
            print(f"{label} EQ Mode set to {active_gesture.upper()}")
        
        eq_state['last_gesture'][label] = active_gesture
    else:
        eq_state['last_gesture'][label] = None
        
    current_mode = eq_state['mode'][label]
    if current_mode != 'none' and (in_inc or in_dec):
        idx_tip = np.array([lm[8].x * w, lm[8].y * h])
        mid_tip = np.array([lm[12].x * w, lm[12].y * h])
        dist = np.linalg.norm(idx_tip - mid_tip)
        min_dist, max_dist = 0, 200  # Generic range
        cc_val = int(np.clip((dist - min_dist) / (max_dist - min_dist) * 127, 0, 127))
        last = eq_state['last'][label]
        
        if abs(cc_val - last) > 2 or not eq_state['active'][label]:
            eq_state['last'][label] = cc_val
            eq_state['active'][label] = True
            ch = 0 if label == 'Left' else 1
            cc_map = {'low': LOW_EQ_CC, 'mid': MID_EQ_CC, 'high': HIGH_EQ_CC}
            control_cc = cc_map[current_mode]
            value_to_send = cc_val if in_inc else (127 - cc_val)
            outport.send(mido.Message('control_change', channel=ch, control=control_cc, value=value_to_send))
            direction = '↑' if in_inc else '↓'
            print(f"{label} {current_mode.capitalize()} EQ {direction} {value_to_send} (ch{ch})")
            set_gesture_msg(f"{label} {current_mode.capitalize()} EQ {direction} {value_to_send} (ch{ch})")
    elif not (in_inc or in_dec):
        eq_state['active'][label] = False

def handle_loop(label, gesture, trackers, outport, lm, frame):
    if lm and frame is not None:
        h, w, _ = frame.shape
        zones = get_eq_zones(label, w, h)
        mode_zone = zones['mode']
        x = lm[0].x * w
        y = lm[0].y * h
        in_mode = (mode_zone[0] <= x <= mode_zone[2] and mode_zone[1] <= y <= mode_zone[3])
        if in_mode:
            return

    # Only trigger for the one_beat_loop gesture (True/False)
    if not gesture:
        trackers['last_gesture'][label] = None
        return
    # Toggle logic: only trigger on rising edge
    if trackers['last_gesture'][label] != gesture:
        trackers['last_gesture'][label] = gesture
        # Toggle state
        state = trackers['state'][label]
        trackers['state'][label] = not state
        ch = 0 if label == 'Left' else 1
        outport.send(mido.Message('note_on', channel=ch,
                                  note=LOOP_NOTE, velocity=127))
        print(f"{label} 1-beat loop {'ON' if not state else 'OFF'} (ch{ch})")
        set_gesture_msg(f"{label} 1-beat loop {'ON' if not state else 'OFF'} (ch{ch})")

def handle_loop2(label, gesture, trackers, outport, lm, frame):
    if lm and frame is not None:
        h, w, _ = frame.shape
        zones = get_eq_zones(label, w, h)
        mode_zone = zones['mode']
        x = lm[0].x * w
        y = lm[0].y * h
        in_mode = (mode_zone[0] <= x <= mode_zone[2] and mode_zone[1] <= y <= mode_zone[3])
        if in_mode:
            return
            
    # Only trigger for the two_beat_loop gesture (True/False)
    if not gesture:
        trackers['last_gesture'][label] = None
        return
    # Toggle logic: only trigger on rising edge
    if trackers['last_gesture'][label] != gesture:
        trackers['last_gesture'][label] = gesture
        # Toggle state
        state = trackers['state'][label]
        trackers['state'][label] = not state
        ch = 0 if label == 'Left' else 1
        outport.send(mido.Message('note_on', channel=ch,
                                  note=LOOP2_NOTE, velocity=127))
        print(f"{label} 2-beat loop {'ON' if not state else 'OFF'} (ch{ch})")
        set_gesture_msg(f"{label} 2-beat loop {'ON' if not state else 'OFF'} (ch{ch})")

def handle_beat_sync(label, _, trackers, outport, frame):
    last_time, sync_order, active, prev_in_button = (
        trackers['last_time'], trackers['sync_order'], trackers['active'], trackers['prev_in_button']
    )
    now = time.time()
    
    # Draw sync toggle buttons as small circles
    h, w, _ = frame.shape
    circle_radius = 40
    y_center = 900
    left_center = (625 + circle_radius, y_center)
    right_center = (w - 625 - circle_radius, y_center)
    
    # Draw circles
    overlay = frame.copy()
    # Colors
    edge_color = (0, 180, 255)  # Orangish yellow (BGR)
    fill_color = (0, 0, 0)      # Black
    alpha_on = 0.9
    alpha_off = 0.2
    font = FONT_FACE
    font_scale = 0.45
    thickness = 1
    text_color = FONT_COLOR
    # For left button
    if not active['Left']:
        cv2.circle(overlay, left_center, circle_radius, fill_color, -1)
        cv2.circle(overlay, left_center, circle_radius, edge_color, 3)
        cv2.addWeighted(overlay, alpha_on, frame, 1-alpha_on, 0, frame)
    else:
        cv2.circle(overlay, left_center, circle_radius, fill_color, -1)
        cv2.circle(overlay, left_center, circle_radius, edge_color, 3)
        cv2.addWeighted(overlay, alpha_off, frame, 1-alpha_off, 0, frame)
    # Draw 'BEAT' and 'SYNC' on separate lines, centered
    beat_text = 'BEAT'
    sync_text = 'SYNC'
    beat_size = cv2.getTextSize(beat_text, font, font_scale, thickness)[0]
    sync_size = cv2.getTextSize(sync_text, font, font_scale, thickness)[0]
    center_x = left_center[0]
    beat_y = left_center[1] - 4
    sync_y = left_center[1] + sync_size[1] + 4
    cv2.putText(frame, beat_text, (center_x - beat_size[0] // 2, beat_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.putText(frame, sync_text, (center_x - sync_size[0] // 2, sync_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    # For right button
    overlay = frame.copy()
    if not active['Right']:
        cv2.circle(overlay, right_center, circle_radius, fill_color, -1)
        cv2.circle(overlay, right_center, circle_radius, edge_color, 3)
        cv2.addWeighted(overlay, alpha_on, frame, 1-alpha_on, 0, frame)
    else:
        cv2.circle(overlay, right_center, circle_radius, fill_color, -1)
        cv2.circle(overlay, right_center, circle_radius, edge_color, 3)
        cv2.addWeighted(overlay, alpha_off, frame, 1-alpha_off, 0, frame)
    beat_size = cv2.getTextSize(beat_text, font, font_scale, thickness)[0]
    sync_size = cv2.getTextSize(sync_text, font, font_scale, thickness)[0]
    center_x = right_center[0]
    beat_y = right_center[1] - 4
    sync_y = right_center[1] + sync_size[1] + 4
    cv2.putText(frame, beat_text, (center_x - beat_size[0] // 2, beat_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.putText(frame, sync_text, (center_x - sync_size[0] // 2, sync_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Check if wrist is in button area
    x = trackers['current_pos'][label][0] * w
    y = trackers['current_pos'][label][1] * h
    in_left = (np.linalg.norm(np.array([x, y]) - np.array(left_center)) <= circle_radius)
    in_right = (np.linalg.norm(np.array([x, y]) - np.array(right_center)) <= circle_radius)
    # Edge trigger logic
    if label == 'Left':
        if in_left and not prev_in_button['Left'] and now - last_time[label] > BEAT_SYNC_DEBOUNCE_TIME:
            if not active['Left']:
                active['Left'] = True
                if 'Left' not in sync_order:
                    sync_order.append('Left')
                print('Left beat sync activated')
                outport.send(mido.Message('note_on', channel=0, note=BEAT_SYNC_NOTE, velocity=127))
            else:
                active['Left'] = False
                if 'Left' in sync_order:
                    sync_order.remove('Left')
                print('Left beat sync deactivated')
                outport.send(mido.Message('note_on', channel=0, note=BEAT_SYNC_NOTE, velocity=0))
            last_time[label] = now
        prev_in_button['Left'] = in_left
    elif label == 'Right':
        if in_right and not prev_in_button['Right'] and now - last_time[label] > BEAT_SYNC_DEBOUNCE_TIME:
            if not active['Right']:
                active['Right'] = True
                if 'Right' not in sync_order:
                    sync_order.append('Right')
                print('Right beat sync activated')
                outport.send(mido.Message('note_on', channel=1, note=BEAT_SYNC_NOTE, velocity=127))
            else:
                active['Right'] = False
                if 'Right' in sync_order:
                    sync_order.remove('Right')
                print('Right beat sync deactivated')
                outport.send(mido.Message('note_on', channel=1, note=BEAT_SYNC_NOTE, velocity=0))
            last_time[label] = now
        prev_in_button['Right'] = in_right
    # Optionally, print which track is leading if both are active
    if active['Left'] and active['Right'] and len(sync_order) == 2:
        ch = 0 if sync_order[0] == 'Left' else 1
        print(f"Beat sync: {sync_order[0]} track leading (ch{ch})")
        set_gesture_msg(f"Beat sync: {sync_order[0]} track leading (ch{ch})")

def handle_stem_toggles(label, lm, h, trackers, outport):
    # Only trigger if back of hand is facing camera
    if is_palm_facing_camera(lm, label):
        trackers['last_gesture'][label] = None
        return
    fingers = count_fingers_up_back(lm, h)
    # Only index up
    if fingers == [True, False, False, False]:
        gesture = 'drums'
        note = STEM_DRUMS_NOTE
    # Index + middle up
    elif fingers == [True, True, False, False]:
        gesture = 'vocals'
        note = STEM_VOCALS_NOTE
    # Robust instrumentals: index down, middle/ring/pinky up (with threshold)
    elif (
        not is_finger_up(lm, 8, 6, h) and
        is_finger_up(lm, 12, 10, h) and
        is_finger_up(lm, 16, 14, h) and
        is_finger_up(lm, 20, 18, h)
    ):
        gesture = 'inst'
        note = STEM_INST_NOTE
    else:
        trackers['last_gesture'][label] = None
        return
    # Toggle logic: only trigger on rising edge
    if trackers['last_gesture'][label] != gesture:
        trackers['last_gesture'][label] = gesture
        # Toggle state
        state = trackers['state'][label][gesture]
        trackers['state'][label][gesture] = not state
        ch = 0 if label == 'Left' else 1
        velocity = 127 if not state else 0  # 127 = mute, 0 = unmute
        outport.send(mido.Message('note_on', channel=ch, note=note, velocity=velocity))
        print(f"{label} hand: {'MUTE' if velocity==127 else 'UNMUTE'} {gesture.upper()} (ch{ch})")
        set_gesture_msg(f"{label} hand: {'MUTE' if velocity==127 else 'UNMUTE'} {gesture.upper()} (ch{ch})")

def draw_stem_bars(frame, stem_state):
    h, w, _ = frame.shape
    bar_height = 18
    bar_width = 80
    gap = 10
    top_margin = 20
    left_margin = 60
    right_margin = 60
    colors = [(255,0,0), (0,255,0), (0,0,255)]  # Blue, Green, Red
    labels = ['Drums', 'Vocals', 'Inst']
    # Left deck
    for i, (stem, color, label) in enumerate(zip(['drums','vocals','inst'], colors, labels)):
        x1 = left_margin + i*(bar_width+gap)
        y1 = top_margin
        x2 = x1 + bar_width
        y2 = y1 + bar_height
        overlay = frame.copy()
        alpha = 0.9 if not stem_state['Left'][stem] else 0.2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        # Text
        font = FONT_FACE
        font_scale = 0.5
        thickness = 1 if not stem_state['Left'][stem] else 1
        text_color = (255,255,255) if not stem_state['Left'][stem] else (200,200,200)
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = x1 + (bar_width - text_size[0]) // 2
        text_y = y1 + (bar_height + text_size[1]) // 2
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    # Right deck
    for i, (stem, color, label) in enumerate(zip(['drums','vocals','inst'], colors, labels)):
        x1 = w - right_margin - (3-i)*(bar_width+gap)
        y1 = top_margin
        x2 = x1 + bar_width
        y2 = y1 + bar_height
        overlay = frame.copy()
        alpha = 0.9 if not stem_state['Right'][stem] else 0.2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        # Text
        font = FONT_FACE
        font_scale = 0.5
        thickness = 1 if not stem_state['Right'][stem] else 1
        text_color = (255,255,255) if not stem_state['Right'][stem] else (200,200,200)
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = x1 + (bar_width - text_size[0]) // 2
        text_y = y1 + (bar_height + text_size[1]) // 2
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# ——— Main ———
def main():
    print("MIDI ports:", mido.get_output_names())
    out = mido.open_output('IAC Driver Bus 1')

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        model_complexity=0
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    pp = {
        'prev_thumb': {'Left':None,'Right':None},
        'last_raw':   {'Left':None,'Right':None},
        'counts':     defaultdict(int),
        'last_time':  defaultdict(lambda:0.0),
        'is_playing': {'Left':False,'Right':False},
        'last_gesture': {'Left': None, 'Right': None}
    }
    loop_st = {
        'last_gesture': {'Left': None, 'Right': None},
        'state': {'Left': False, 'Right': False}
    }
    loop2_st = {
        'last_gesture': {'Left': None, 'Right': None},
        'state': {'Left': False, 'Right': False}
    }
    beat_sync_st = {
        'last_raw':  {'Left':False,'Right':False},
        'counts':    defaultdict(int),
        'last_time': defaultdict(lambda:0.0),
        'sync_order': [],
        'current_pos': {'Left': (0,0), 'Right': (0,0)},
        'active': {'Left': False, 'Right': False},
        'prev_in_button': {'Left': False, 'Right': False}
    }
    stem_trackers = {
        'last_gesture': {'Left': None, 'Right': None},
        'state': {
            'Left': {'drums': False, 'vocals': False, 'inst': False},
            'Right': {'drums': False, 'vocals': False, 'inst': False}
        }
    }

    vol = {'last':{'Left':64,'Right':64}, 'threshold':VOL_THRESHOLD}
    tmp = {'last':{'Left':64,'Right':64}, 'threshold':TEMPO_THRESHOLD}
    df  = {'last':{'Left':False,'Right':False}}
    eq_st = {
        'mode': {'Left': 'none', 'Right': 'none'},
        'last': {'Left': -1, 'Right': -1},
        'active': {'Left': False, 'Right': False},
        'last_gesture': {'Left': None, 'Right': None}
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        draw_stem_bars(frame, stem_trackers['state'])

        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = hd.classification[0].label
                palm_forward = is_palm_facing_camera(lm.landmark, label)
                open_flag   = is_hand_open(lm.landmark)
                closed_flag = is_hand_closed(lm.landmark, h)

                # detect gestures
                idx_up = is_index_up(lm.landmark,h)
                one_beat_loop = is_index_and_middle_up(lm.landmark, h)
                two_beat_loop = is_middle_ring_pinky_up(lm.landmark, h)
                
                eq_gestures = {
                    'index_up': idx_up,
                    'one_beat_loop': one_beat_loop,
                    'two_beat_loop': two_beat_loop
                }

                # Store current hand position for beat sync
                beat_sync_st['current_pos'][label] = (lm.landmark[0].x, lm.landmark[0].y)

                # Consolidated EQ handler
                handle_eq(label, open_flag, lm.landmark, out, frame, eq_st, eq_gestures, palm_forward)

                # Other handlers are conditional on the gesture not being consumed by EQ
                play_pause_gesture = idx_up and not one_beat_loop
                handle_playpause(label, 'IndexUp' if play_pause_gesture and palm_forward else None, pp, out, palm_forward, lm.landmark, frame)
                
                if palm_forward:
                    handle_loop(label, one_beat_loop, loop_st, out, lm.landmark, frame)
                    handle_loop2(label, two_beat_loop, loop2_st, out, lm.landmark, frame)

                handle_beat_sync(label, None, beat_sync_st, out, frame)
                handle_stem_toggles(label, lm.landmark, h, stem_trackers, out)
                if not df['last'][label]:
                    handle_volume(label,open_flag,lm.landmark,vol,out,frame)
                    handle_tempo(label,closed_flag,lm.landmark,tmp,out,frame)
                    
                mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)

        # Draw the current gesture message at the top/middle
        if CURRENT_GESTURE_MSG['msg']:
            font = FONT_FACE
            font_scale = 1.1
            thickness = 2
            text = CURRENT_GESTURE_MSG['msg']
            h, w, _ = frame.shape
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 60
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255,255,0), thickness, cv2.LINE_AA)

        cv2.imshow("DJ Controller", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    out.close()

if __name__ == "__main__":
    main()
