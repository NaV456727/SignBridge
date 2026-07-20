import cv2
import textwrap

from src.ui.colors import *

# ----------------------------------------------------------------------
# Colors not assumed to already exist in colors.py.
# colors.py is expected to provide: BACKGROUND, PANEL, PANEL_BORDER,
# WHITE, GRAY, GREEN. Everything else needed for the confidence
# scale and progress-bar track is defined locally below (all BGR,
# since these are OpenCV colors).
# ----------------------------------------------------------------------
AMBER = (0, 191, 255)        # 60-79% confidence
RED = (60, 60, 235)          # <60% confidence
TRACK_BG = (55, 55, 55)      # progress bar empty track
CARD_MUTED = (30, 30, 30)    # nested / secondary card fill

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA


class Overlay:
    """
    Draws the SignBridge HUD entirely with OpenCV primitives.

    Layout:
        - Header bar: app name/version (left), LIVE indicator + FPS (right)
        - Webcam feed: largest element, left side
        - Sidebar (right): "Current Sign" card + "Predictions" card
        - Sentence panel: full-width strip along the bottom
    """

    def __init__(self):

        self.header_height = 64
        self.sidebar_width = 400
        self.sentence_height = 140
        self.margin = 20
        self.radius = 14

    # ==========================================================
    # Public API — signature is intentionally unchanged
    # ==========================================================

    def draw(
        self,
        frame,
        prediction,
        confidence,
        sentence,
        fps,
        hand_count,
        mode="ASL",
        top_predictions=None,
    ):

        # Defensive normalization: never let None reach a text-drawing
        # call. hand_count / mode are accepted (API contract) but are
        # intentionally not rendered anywhere in this layout.
        prediction = prediction if prediction else ""
        sentence = sentence if sentence else ""
        confidence = confidence if confidence else 0
        fps = fps if fps else 0

        frame = cv2.copyMakeBorder(
            frame,
            self.header_height + 10,
            self.sentence_height + 10,
            10,
            self.sidebar_width + 10,
            cv2.BORDER_CONSTANT,
            value=BACKGROUND,
        )

        h, w = frame.shape[:2]

        webcam_left = 10
        webcam_top = self.header_height + 10
        webcam_right = w - self.sidebar_width - 20
        webcam_bottom = h - self.sentence_height - 20

        sidebar_left = webcam_right + 20
        sidebar_right = w - 20

        self.draw_header(frame, fps)
        self.draw_webcam_border(frame, webcam_left, webcam_top, webcam_right, webcam_bottom)
        self.draw_sidebar(
            frame,
            sidebar_left,
            webcam_top,
            sidebar_right,
            webcam_bottom,
            prediction,
            confidence,
            top_predictions=top_predictions,
        )
        self.draw_sentence_panel(frame, sentence)

        # Outer window frame
        cv2.rectangle(frame, (5, 5), (w - 5, h - 5), PANEL_BORDER, 1, AA)

        return frame

    # ==========================================================
    # Low-level drawing helpers
    # ==========================================================

    def rounded_rect(self, img, pt1, pt2, radius, color, thickness=-1):
        """Filled or outlined rounded rectangle, built from OpenCV primitives."""

        x1, y1 = pt1
        x2, y2 = pt2
        radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

        if thickness < 0:
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1, AA)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1, AA)
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, AA)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, AA)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, AA)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, AA)
        else:
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, AA)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, AA)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, AA)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, AA)
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, AA)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, AA)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, AA)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, AA)

    def panel(self, img, pt1, pt2, radius=None, fill=None, border=None, alpha=0.94):
        """Translucent rounded panel with an optional border on top."""

        if radius is None:
            radius = self.radius

        if fill is not None:
            layer = img.copy()
            self.rounded_rect(layer, pt1, pt2, radius, fill, -1)
            cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)

        if border is not None:
            self.rounded_rect(img, pt1, pt2, radius, border, 1)

    def draw_card(self, frame, pt1, pt2, fill=PANEL, border=PANEL_BORDER):
        """Standard sidebar/panel card: rounded, translucent, bordered."""

        self.panel(frame, pt1, pt2, radius=self.radius, fill=fill, border=border)

    def draw_centered_text(self, frame, text, center_x, y, font, scale, color, thickness):
        """Draws text horizontally centered on center_x, baseline at y."""

        size = cv2.getTextSize(text, font, scale, thickness)[0]
        x = int(center_x - size[0] / 2)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, AA)
        return size

    def draw_progress_bar(self, frame, left, right, top, bottom, percent, color):
        """Rounded confidence bar with a filled portion proportional to percent."""

        bar_radius = max(2, (bottom - top) // 2)

        self.rounded_rect(frame, (left, top), (right, bottom), bar_radius, TRACK_BG, -1)

        percent = max(0, min(percent, 100))
        fill_width = int((right - left) * percent / 100)

        if fill_width > bar_radius * 2:
            self.rounded_rect(frame, (left, top), (left + fill_width, bottom), bar_radius, color, -1)
        elif fill_width > 0:
            cv2.circle(frame, (left + bar_radius, (top + bottom) // 2), bar_radius, color, -1, AA)

        self.rounded_rect(frame, (left, top), (right, bottom), bar_radius, PANEL_BORDER, 1)

    def confidence_color(self, confidence):
        """Green / amber / red scale used for both the letter and the bar."""

        if confidence >= 80:
            return GREEN
        if confidence >= 60:
            return AMBER
        return RED

    # ==========================================================
    # Header
    # ==========================================================

    def draw_header(self, frame, fps):

        h, w = frame.shape[:2]
        baseline_y = self.header_height // 2 + 9

        cv2.putText(frame, "SignBridge", (24, baseline_y), FONT_BOLD, 0.9, WHITE, 1, AA)

        name_width = cv2.getTextSize("SignBridge", FONT_BOLD, 0.9, 1)[0][0]
        cv2.putText(frame, "v1.0", (24 + name_width + 12, baseline_y), FONT, 0.55, GRAY, 1, AA)

        # FPS, right-aligned
        fps_text = f"{int(fps)} FPS"
        fps_color = GREEN if fps >= 20 else (AMBER if fps >= 10 else RED)
        fps_size = cv2.getTextSize(fps_text, FONT, 0.65, 2)[0]
        fps_x = w - 24 - fps_size[0]
        cv2.putText(frame, fps_text, (fps_x, baseline_y), FONT, 0.65, fps_color, 2, AA)

        # LIVE indicator, left of the FPS readout
        live_text = "LIVE"
        live_size = cv2.getTextSize(live_text, FONT, 0.6, 2)[0]
        live_text_x = fps_x - 28 - live_size[0]
        dot_x = live_text_x - 14
        dot_y = baseline_y - 6

        cv2.circle(frame, (dot_x, dot_y), 5, GREEN, -1, AA)
        cv2.putText(frame, live_text, (live_text_x, baseline_y), FONT, 0.6, GREEN, 2, AA)

        cv2.line(frame, (0, self.header_height), (w, self.header_height), PANEL_BORDER, 1, AA)

    # ==========================================================
    # Webcam
    # ==========================================================

    def draw_webcam_border(self, frame, left, top, right, bottom):
        """Subtle outline only — never covers the webcam image itself."""

        self.rounded_rect(frame, (left, top), (right, bottom), self.radius, PANEL_BORDER, 1)

    # ==========================================================
    # Sidebar
    # ==========================================================

    def draw_sidebar(self, frame, left, top, right, bottom, prediction, confidence, top_predictions):

        gap = self.margin
        total_height = bottom - top

        prediction_card_bottom = top + int(total_height * 0.62)
        list_card_top = prediction_card_bottom + gap

        self.draw_prediction_card(frame, left, top, right, prediction_card_bottom, prediction, confidence)

        # No alternate-prediction data is available from the draw() API,
        # so this always renders the graceful placeholder rather than
        # fabricating candidate letters/percentages.
        self.draw_prediction_list(frame, left, list_card_top, right, bottom, alternates=top_predictions)

    # ==========================================================
    # Current Sign Card
    # ==========================================================

    def draw_prediction_card(self, frame, left, top, right, bottom, prediction, confidence):

        self.draw_card(frame, (left, top), (right, bottom))

        pad = 24
        center_x = (left + right) // 2

        title_y = top + 38
        self.draw_centered_text(frame, "CURRENT SIGN", center_x, title_y, FONT, 0.65, GRAY, 2)

        divider_y = title_y + 20
        cv2.line(frame, (left + pad, divider_y), (right - pad, divider_y), PANEL_BORDER, 1, AA)

        inner_top = divider_y
        inner_bottom = bottom - pad
        inner_height = inner_bottom - inner_top

        color = self.confidence_color(confidence)
        display = prediction.strip() if prediction.strip() != "" else "--"

        letter_y = inner_top + int(inner_height * 0.52)
        letter_scale = 4.3
        letter_thickness = 7

        self.draw_centered_text(frame, display, center_x, letter_y, FONT_BOLD, letter_scale, color, letter_thickness)

        conf_text = f"{confidence:.1f}%"
        conf_y = letter_y + int(inner_height * 0.14)
        self.draw_centered_text(frame, conf_text, center_x, conf_y, FONT, 0.85, WHITE, 2)

        bar_top = conf_y + 22
        bar_bottom = bar_top + 22
        self.draw_progress_bar(frame, left + pad, right - pad, bar_top, bar_bottom, confidence, color)

    # ==========================================================
    # Prediction Card
    # ==========================================================

    def draw_prediction_list(self, frame, left, top, right, bottom, alternates=None):
        """
        Renders up to three alternate predictions as (label, percent)
        tuples, e.g. [("A", 96.0), ("R", 3.0), ("U", 1.0)].

        If `alternates` is None or empty, shows a clean placeholder
        instead of fabricating data — the draw() API currently has no
        parameter carrying this information.
        """

        self.draw_card(frame, (left, top), (right, bottom))

        pad = 24
        center_x = (left + right) // 2

        title_y = top + 34
        self.draw_centered_text(frame, "PREDICTIONS", center_x, title_y, FONT, 0.6, GRAY, 2)

        divider_y = title_y + 18
        cv2.line(frame, (left + pad, divider_y), (right - pad, divider_y), PANEL_BORDER, 1, AA)

        content_top = divider_y + 10
        content_bottom = bottom - pad

        if not alternates:
            placeholder_y = content_top + (content_bottom - content_top) // 2 + 6
            self.draw_centered_text(
                frame,
                "No alternate predictions",
                center_x,
                placeholder_y,
                FONT,
                0.55,
                GRAY,
                1,
            )
            return

        rows = alternates[:3]
        row_height = (content_bottom - content_top) // max(1, len(rows))

        for i, (label, percent) in enumerate(rows):

            row_top = content_top + i * row_height
            row_center_y = row_top + row_height // 2 + 6

            cv2.putText(frame, str(label), (left + pad, row_center_y), FONT, 0.7, WHITE, 2, AA)

            percent_text = f"{percent:.0f}%"
            size = cv2.getTextSize(percent_text, FONT, 0.6, 2)[0]
            cv2.putText(
                frame,
                percent_text,
                (right - pad - size[0], row_center_y),
                FONT,
                0.6,
                self.confidence_color(percent),
                2,
                AA,
            )

    # ==========================================================
    # Sentence Panel
    # ==========================================================

    def draw_sentence_panel(self, frame, sentence):

        h, w = frame.shape[:2]

        top = h - self.sentence_height
        bottom = h - 10
        left = 10
        right = w - 10

        self.draw_card(frame, (left, top), (right, bottom))

        pad = 24

        cv2.putText(frame, "Sentence", (left + pad, top + 36), FONT, 0.7, GRAY, 2, AA)

        length_text = f"{len(sentence)} chars"
        size = cv2.getTextSize(length_text, FONT, 0.5, 1)[0]
        cv2.putText(
            frame,
            length_text,
            (right - pad - size[0], top + 36),
            FONT,
            0.5,
            GRAY,
            1,
            AA,
        )

        divider_y = top + 52
        cv2.line(frame, (left + pad, divider_y), (right - pad, divider_y), PANEL_BORDER, 1, AA)

        display = sentence if sentence.strip() != "" else "Waiting for input..."
        text_color = WHITE if sentence.strip() != "" else GRAY

        chars_per_line = max(1, (right - left - pad * 2) // 16)
        wrapped = textwrap.wrap(display, width=chars_per_line) or [""]

        y = divider_y + 34
        for line in wrapped[:2]:
            cv2.putText(frame, line, (left + pad, y), FONT, 0.85, text_color, 2, AA)
            y += 34
