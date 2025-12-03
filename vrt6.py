import cv2
import numpy as np
import math


class HandSafetySystem:
    def __init__(self):
        # 1. Initialize Camera
        self.cap = cv2.VideoCapture(0)

        # 2. Virtual Object (The Danger Zone)
        self.danger_zone_rect = None

        # 3. State Constants
        self.STATE_SAFE = 0
        self.STATE_WARNING = 1
        self.STATE_DANGER = 2

        # 4. TRACKING: AUTO-START WITH PRESET
        # We start as "True" so you don't need to press anything
        self.calibrated = True

        # Generic Skin Tone Range (Works for most lighting)
        # Hue: 0-25 (Skin/Orange), Saturation: 30-255, Value: 60-255
        self.lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_hsv = np.array([25, 255, 255], dtype=np.uint8)

        print("[SYSTEM] System Armed. Tracking started automatically.")

    def process_frame(self, frame):
        """
        Detects the hand using Color Segmentation + Convex Hull
        """
        # Blur to remove camera noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Color Masking
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Clean up the mask (fill holes)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hull_points = None

        if len(contours) > 0:
            # Find the largest object (The Hand)
            max_contour = max(contours, key=cv2.contourArea)

            # Filter out small noise
            if cv2.contourArea(max_contour) > 3000:
                # Calculate Convex Hull (The outline of the hand)
                hull_points = cv2.convexHull(max_contour)

        return hull_points

    def calculate_logic(self, hull, rect):
        """
        Scans EVERY point on the hand's outline to see if *any* part
        is touching the Danger Zone.
        """
        if hull is None:
            return self.STATE_SAFE, 0, (0, 0), (0, 0)

        rx, ry, rw, rh = rect

        min_dist = 99999
        closest_hand_point = (0, 0)
        closest_box_point = (0, 0)

        # Loop through every point on the hand's hull
        for point in hull:
            px, py = point[0]

            # Find the closest point on the Box to this specific part of the hand
            cl_x = max(rx, min(px, rx + rw))
            cl_y = max(ry, min(py, ry + rh))

            # Calculate distance
            dist = math.sqrt((px - cl_x) ** 2 + (py - cl_y) ** 2)

            # Update if this is the new closest point
            if dist < min_dist:
                min_dist = dist
                closest_hand_point = (px, py)
                closest_box_point = (cl_x, cl_y)

        # Logic Thresholds
        if min_dist <= 10:
            return self.STATE_DANGER, min_dist, closest_hand_point, closest_box_point
        elif min_dist < 150:
            return self.STATE_WARNING, min_dist, closest_hand_point, closest_box_point
        else:
            return self.STATE_SAFE, min_dist, closest_hand_point, closest_box_point

    def run(self):
        print("--- ZERO-CLICK SAFETY DEMO ---")
        print("System is running. Press 'q' to Quit.")

        while True:
            ret, frame = self.cap.read()
            if not ret: break

            # Mirror the frame
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Define Danger Zone (Right side)
            if self.danger_zone_rect is None:
                self.danger_zone_rect = (int(w * 0.65), int(h * 0.25), 200, 200)

            ox, oy, ow, oh = self.danger_zone_rect

            # 1. Get Hand Hull
            hull = self.process_frame(frame)

            # 2. Calculate Distance (Full Hand)
            state, dist, hand_pt, box_pt = self.calculate_logic(hull, self.danger_zone_rect)

            # 3. Visualization Setup
            color_map = {
                self.STATE_SAFE: (0, 255, 0),  # Green
                self.STATE_WARNING: (0, 255, 255),  # Yellow
                self.STATE_DANGER: (0, 0, 255)  # Red
            }
            theme_color = color_map[state]

            # Draw Zone
            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), theme_color, 3)
            cv2.putText(frame, "VIRTUAL OBJECT", (ox, oy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme_color, 2)

            # Draw Hand
            if hull is not None:
                # Draw the outline of the hand
                cv2.drawContours(frame, [hull], -1, (255, 255, 0), 2)

                # Draw the interaction line
                if state != self.STATE_SAFE:
                    cv2.line(frame, hand_pt, box_pt, (255, 255, 255), 2)
                    cv2.circle(frame, hand_pt, 8, (0, 0, 255), -1)  # Red dot on the closest part of hand
                    cv2.circle(frame, box_pt, 5, (255, 255, 255), -1)  # White dot on the box

                    cv2.putText(frame, f"{int(dist)}px", (hand_pt[0] + 10, hand_pt[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 4. On-Screen Alerts
            if state == self.STATE_DANGER:
                # Red Border
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
                # Text
                cv2.putText(frame, "DANGER DANGER", (50, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
            elif state == self.STATE_WARNING:
                cv2.putText(frame, "WARNING: APPROACHING", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "STATE: SAFE", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Hand Safety Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandSafetySystem()
    app.run()