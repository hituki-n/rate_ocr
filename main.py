import json
from datetime import datetime
import re
import logging

from PIL import Image
import pandas as pd
import numpy as np
import cv2
import pyocr
from functools import reduce, partial

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFormLayout, QSpinBox, QCheckBox, \
    QComboBox, QLineEdit
import pyqtgraph as pg
import sys
import os

import i18n

i18n.load_path.append('locales')
i18n.set('filename_format', '{locale}.{format}')
i18n.set('fallback', 'en')
_ = i18n.t

APP_NAME = "Puyo Rate Detector"
APP_VER = "20220101_1"
OPTION_FILE = "option.json"


def main():
    i18n.set('locale', app_option['app_locale'])
    logging.getLogger().setLevel(logging.INFO)
    app = QtWidgets.QApplication(sys.argv)
    main_window = RateDetectionWindow()
    main_window.show()
    sys.exit(app.exec_())


if not os.path.exists(OPTION_FILE):
    logging.error("not found option file(%s)" % OPTION_FILE)
    exit()
with open(OPTION_FILE, encoding='utf-8', mode='r') as ofile:
    app_option = json.load(ofile)


class RateDetectionWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(640, 480)
        self.setStyleSheet("background-color: black")
        self.setWindowTitle(APP_NAME)

        self.rate_detector = RateDetector()
        self.option_window = None

        main_widget = QWidget()
        vbox = QVBoxLayout()
        header_layout = QHBoxLayout()
        vbox.addLayout(header_layout)
        main_widget.setLayout(vbox)
        self.setCentralWidget(main_widget)

        font = QtGui.QFont()
        font.setPixelSize(35)
        pen = pg.mkPen(color=(152, 230, 152), width=6)

        self.wl_label = QLabel()
        self.set_wl()
        self.wl_label.setAlignment(QtCore.Qt.AlignCenter)
        self.wl_label.setStyleSheet("color: ivory")
        self.wl_label.setFont(font)
        header_layout.addWidget(self.wl_label)

        option_button = QPushButton()
        option_button.setFixedSize(40, 40)
        option_button.setStyleSheet("background-image:url(icons/option.png);background-position:center")
        option_button.clicked.connect(self.open_option_window)
        header_layout.addWidget(option_button)

        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground('black')
        self.graphWidget.getPlotItem().hideAxis('bottom')
        left_axis = self.graphWidget.getPlotItem().getAxis('left')
        left_axis.setStyle(tickFont=font)
        left_axis.setTextPen('ivory')
        left_axis.setPen('yellow')
        self.graphWidget.getPlotItem().showGrid(y=True)
        vbox.addWidget(self.graphWidget)

        self.my_rate_line = self.graphWidget.plot(pen=pen)
        self.opponent_plots = []
        self.show_chart()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(app_option['check_interval'])
        self.timer.timeout.connect(self.check_rate)
        self.timer.start()

    def check_rate(self):
        opening, ending = self.rate_detector.check_rate()
        if opening:
            self.show_chart()
        if ending:
            self.set_wl()

    def set_wl(self):
        self.wl_label.setText(_("chart.wl",
                                win=self.rate_detector.history_manager.win_cnt,
                                lose=self.rate_detector.history_manager.lose_cnt))

    def open_option_window(self):
        self.timer.stop()
        if self.option_window is None:
            self.option_window = OptionWindow(main_window=self)
        self.option_window.show()

    def show_chart(self):
        game_history = self.rate_detector.history_manager.game_history
        self.my_rate_line.setData(range(len(game_history)), game_history["my_rate"].tolist())
        if app_option['plot_opponent_rate']:
            for i in range(len(self.opponent_plots), len(game_history)):
                data = game_history.iloc[i]
                self.opponent_plots.append(self.plot_opponent_rate(i, data["opponent_rate"], data["my_rate"]))

    def plot_opponent_rate(self, i, opponent_rate, my_rate):
        green = int((my_rate - opponent_rate + 500) * 255 / 1000)
        green = green if 0 <= green <= 255 else 0 if green < 0 else 255
        plot = self.graphWidget.plot([i], [opponent_rate], pen=None, symbol='o', symbolBrush=(255, green, 0))
        plot.setAlpha(0.85, False)
        return plot

    def apply_updated_option(self):
        i18n.set('locale', app_option['app_locale'])

        self.rate_detector.history_manager.load_history_file()
        self.my_rate_line.clear()
        for opponent_plot in self.opponent_plots:
            opponent_plot.clear()
        self.opponent_plots.clear()
        self.show_chart()

        self.set_wl()


class OptionWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        layout = QFormLayout()
        self.video_source_window = None
        self.crop_area_window = None
        self.setWindowTitle(APP_NAME + " --option setting--")

        version_label = QLabel(_("option.app_ver"))
        version_input = QLabel(APP_VER)
        layout.addRow(version_label, version_input)
        self.setLayout(layout)

        app_locale = QLabel(_("option.app_locale"))
        self.app_locale = QComboBox()
        locales = {"ja": "日本語", "en": "English"}
        for locale_key in locales.keys():
            self.app_locale.addItem(locales[locale_key], userData=locale_key)
        self.app_locale.setCurrentText(locales[app_option['app_locale']])
        layout.addRow(app_locale, self.app_locale)

        interval_label = QLabel(_("option.check_interval"))
        self.interval_input = QSpinBox()
        self.interval_input.setMaximum(500)
        self.interval_input.setValue(app_option['check_interval'])
        layout.addRow(interval_label, self.interval_input)

        period_label = QLabel((_("option.num_first_plot")))
        self.period_input = QSpinBox()
        self.period_input.setMaximum(10000)
        self.period_input.setValue(app_option['chart_period'])
        layout.addRow(period_label, self.period_input)

        plot_opponent_label = QLabel((_("option.plot_opponent_rate")))
        self.plot_opponent_input = QCheckBox()
        self.plot_opponent_input.setChecked(app_option['plot_opponent_rate'] == "True")
        layout.addRow(plot_opponent_label, self.plot_opponent_input)

        video_source_label = QLabel((_("option.video_source")))
        self.video_source_input = QSpinBox()
        self.video_source_input.setValue(app_option['video_source'])
        layout.addRow(video_source_label, self.video_source_input)

        self.video_source_button = QPushButton()
        self.video_source_button.setFixedSize(18, 18)
        self.video_source_button.setStyleSheet("background-image:url(icons/eye.png);background-position:center")
        self.video_source_button.clicked.connect(self.open_video_source_window)
        layout.addRow("", self.video_source_button)

        crop_area_label = QLabel((_("option.crop_area")))
        self.crop_area_button = QPushButton()
        self.crop_area_button.setFixedSize(18, 18)
        self.crop_area_button.setStyleSheet("background-image:url(icons/eye.png);background-position:center")
        self.crop_area_button.clicked.connect(self.open_crop_area_window)
        layout.addRow(crop_area_label, self.crop_area_button)

        user_name_label = QLabel((_("option.user_name_candidate")))
        self.user_name_input = QLineEdit()
        self.user_name_input.setText(reduce(lambda a, b: a+' '+b, app_option['user_name']) if app_option['user_name']
                                     else '')
        self.user_name_input.setMinimumWidth(500)
        layout.addRow(user_name_label, self.user_name_input)

        ocr_language_label = QLabel((_("option.ocr_language")))
        self.ocr_language_input = QComboBox()
        self.ocr_language_input.addItems(pyocr.get_available_tools()[0].get_available_languages())
        self.ocr_language_input.setCurrentText(app_option['ocr_language'])
        layout.addRow(ocr_language_label, self.ocr_language_input)

        record_file_label = QLabel((_("option.record_file")))
        self.record_file_input = QLineEdit()
        self.record_file_input.setText(app_option['record_file'])
        layout.addRow(record_file_label, self.record_file_input)

        save_button = QPushButton(_("option.save"))
        save_button.setStyleSheet("background-color:aqua")
        save_button.clicked.connect(self.save_option)
        layout.addRow(save_button)

    def closeEvent(self, event):
        self.main_window.option_window = None
        self.main_window.rate_detector.reset_video_source()
        self.main_window.timer.start()

    def save_option(self):
        self.close()
        app_option["saved_version"] = APP_VER
        app_option["lower_rate"] = 1000
        app_option["upper_rate"] = 4000
        app_option["capture_width"] = 1280
        app_option["capture_height"] = 720
        app_option["capture_frame_rate"] = 30

        app_option["app_locale"] = self.app_locale.currentData()
        app_option["check_interval"] = self.interval_input.value()
        app_option["chart_period"] = self.period_input.value()
        app_option["plot_opponent_rate"] = "True" if self.plot_opponent_input.isChecked() else ""
        app_option["video_source"] = self.video_source_input.value()
        app_option["user_name"] = self.user_name_input.text().split()
        app_option["ocr_language"] = self.ocr_language_input.currentText()
        app_option["record_file"] = self.record_file_input.text()

        with open(OPTION_FILE, mode='w', encoding='utf-8') as file:
            json.dump(app_option, file, ensure_ascii=False, indent=4)
        self.main_window.apply_updated_option()

    def open_video_source_window(self):
        if self.video_source_window is None:
            self.video_source_window = VideoSourceWindow(parent=self)
        self.video_source_window.show()

    def apply_new_video_source(self, video_source_id):
        self.video_source_input.setValue(video_source_id)

    def open_crop_area_window(self):
        if self.crop_area_window is None:
            self.crop_area_window = CropAreaWindow(parent=self)
        self.crop_area_window.show()

    def apply_new_crop_area(self, left_rate_area, left_name_area, right_rate_area, right_name_area, end_rate_area):
        app_option["left_rate_area"] = left_rate_area
        app_option["left_name_area"] = left_name_area
        app_option["right_rate_area"] = right_rate_area
        app_option["right_name_area"] = right_name_area
        app_option["end_rate_area"] = end_rate_area


class VideoSourceDisplay(QLabel):
    def __init__(self, video_source_id=None):
        if video_source_id is None:
            video_source_id = app_option['video_source']
        super().__init__()

        self.display_width = 640
        self.display_height = 360
        self.resize(self.display_width, self.display_height)

        self.thread = self.VideoThread(current_video_id=video_source_id)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height)
        return QPixmap.fromImage(p)

    class VideoThread(QThread):
        change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)

        def __init__(self, current_video_id):
            super().__init__()
            self.cap = cv2.VideoCapture(current_video_id)
            RateDetector.default_video_setting(self.cap)

        def run(self):
            while True:
                ret, cv_img = self.cap.read()
                if ret:
                    self.change_pixmap_signal.emit(cv_img)

        def quit(self):
            self.cap.release()


class CropAreaWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.position_inputs = dict()
        self.setWindowTitle(APP_NAME + " --crop area setting--")

        vbox = QVBoxLayout()
        self.setLayout(vbox)

        vbox.addWidget(QLabel(_("crop.exploration")))

        label_texts = [_("crop.left_rate"), _("crop.left_name"),
                       _("crop.right_rate"), _("crop.right_name"), _("crop.end_rate")]
        option_keys = ["left_rate_area", "left_name_area", "right_rate_area", "right_name_area", "end_rate_area"]
        for label_text, option_key in zip(label_texts, option_keys):
            layout = self._generate_position_group(label_text, option_key)
            vbox.addLayout(layout)

        self.video_display = self.VideoSourceInteractiveDisplay(self.parent.video_source_input.value())
        vbox.addWidget(self.video_display)

        confirm_button = QPushButton(_("crop.confirm"))
        confirm_button.setStyleSheet("background-color:aqua")
        confirm_button.clicked.connect(self.save_crop_area)
        vbox.addWidget(confirm_button)

    def closeEvent(self, event):
        self.video_display.thread.quit()
        self.parent.crop_area_window = None

    def _generate_position_group(self, label_text, area_key):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        inputs = self._generate_position_inputs(area_key)
        self.position_inputs[area_key] = inputs
        layout.addWidget(label)
        for key in ['x', 'y', 'w', 'h']:
            layout.addWidget(inputs[key])

        confirm = QPushButton()
        confirm.setText(_("crop.apply"))
        confirm.setFixedWidth(60)
        confirm.clicked.connect(partial(self.apply_position_group, inputs))
        layout.addWidget(confirm)

        return layout

    def _generate_position_inputs(self, area_key):
        position_inputs = dict()
        area = app_option[area_key]
        size_keys = ['capture_width', 'capture_height', 'capture_width', 'capture_height']
        position_keys = ['x', 'y', 'w', 'h']
        for size_key, position_key in zip(size_keys, position_keys):
            position_input = self.PositionInput(parent=self, area_key=area_key)
            position_input.setMaximum(app_option[size_key])
            position_input.setValue(area[position_key])
            position_input.setToolTip(_("crop.tooltip_"+position_key))
            position_inputs[position_key] = position_input

        return position_inputs

    class PositionInput(QSpinBox):
        def __init__(self, parent, area_key):
            super().__init__()
            self.area_key = area_key
            self.parent = parent

        def focusInEvent(self, event):
            area_info = self.parent.position_inputs[self.area_key]
            self.parent.video_display.current_begin_point = (area_info['x'].value(), area_info['y'].value())
            self.parent.video_display.current_end_point = (area_info['x'].value() + area_info['w'].value(),
                                                           area_info['y'].value() + area_info['h'].value())

    def apply_position_group(self, inputs):
        if self.video_display.drawing_begin_point is not None:
            start_x = min(self.video_display.drawing_begin_point[0], self.video_display.drawing_end_point[0])
            start_y = min(self.video_display.drawing_begin_point[1], self.video_display.drawing_end_point[1])
            w = abs(self.video_display.drawing_begin_point[0] - self.video_display.drawing_end_point[0])
            h = abs(self.video_display.drawing_begin_point[1] - self.video_display.drawing_end_point[1])
            inputs['x'].setValue(start_x)
            inputs['y'].setValue(start_y)
            inputs['w'].setValue(w)
            inputs['h'].setValue(h)
        else:
            logging.warning('need to draw rectangle')
        self.video_display.current_begin_point = self.video_display.drawing_begin_point
        self.video_display.current_end_point = self.video_display.drawing_end_point
        self.video_display.drawing_begin_point = None
        self.video_display.drawing_end_point = None

    def save_crop_area(self):
        self.close()
        option_keys = ["left_rate_area", "left_name_area", "right_rate_area", "right_name_area", "end_rate_area"]
        areas = []
        for key in option_keys:
            areas.append(self.convert_to_dict(self.position_inputs[key]))
        self.parent.apply_new_crop_area(*areas)

    @staticmethod
    def convert_to_dict(inputs):
        area = dict()
        for e in ['x', 'y', 'w', 'h']:
            area[e] = inputs[e].value()
        return area

    class VideoSourceInteractiveDisplay(VideoSourceDisplay):
        def __init__(self, video_source_id):
            super().__init__(video_source_id=video_source_id)
            self.setFixedSize(640, 360)
            self.drawing_begin_point = None
            self.drawing_end_point = None
            self.drawing_color = (0, 0, 255)
            self.current_begin_point = None
            self.current_end_point = None
            self.current_color = (0, 255, 0)
            self.scale = app_option["capture_width"] / 640

        def convert_cv_qt(self, cv_img):
            if self.drawing_begin_point is not None:
                cv2.rectangle(cv_img, self.drawing_begin_point, self.drawing_end_point, self.drawing_color, 3)
            if self.current_begin_point is not None:
                cv2.rectangle(cv_img, self.current_begin_point, self.current_end_point, self.current_color, 3)

            return super().convert_cv_qt(cv_img)

        def mousePressEvent(self, event):
            self.drawing_begin_point = self.scaled_point(event.pos())
            self.drawing_end_point = self.scaled_point(event.pos())
            self.update()

        def mouseMoveEvent(self, event):
            self.drawing_end_point = self.scaled_point(event.pos())
            self.update()

        def mouseReleaseEvent(self, event):
            self.drawing_end_point = self.scaled_point(event.pos())
            self.update()

        def scaled_point(self, pos):
            return int(self.scale * pos.x()), int(self.scale * pos.y())


class VideoSourceWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle(APP_NAME + " --video source setting--")
        vbox = QVBoxLayout()
        self.setLayout(vbox)

        self.display_width = 640
        self.display_height = 360

        exploration = QLabel(_("video_source.exploration"))
        vbox.addWidget(exploration)

        self.video_source_input = QSpinBox()
        self.video_source_input.setValue(parent.video_source_input.value())
        self.video_source_input.textChanged.connect(self.change_video_source)
        vbox.addWidget(self.video_source_input)

        self.image_label = VideoSourceDisplay(self.video_source_input.value())
        vbox.addWidget(self.image_label)

        confirm_button = QPushButton(_("video_source.select"))
        confirm_button.setStyleSheet("background-color:aqua")
        confirm_button.clicked.connect(self.apply_video_source)
        vbox.addWidget(confirm_button)

    def closeEvent(self, event):
        self.image_label.thread.quit()
        self.parent.video_source_window = None

    def apply_video_source(self):
        self.close()
        self.parent.apply_new_video_source(self.video_source_input.value())

    def change_video_source(self):
        self.image_label.thread.cap.release()
        self.image_label.thread.cap = cv2.VideoCapture(self.video_source_input.value())


class RateDetector:
    def __init__(self):
        self.ocr_engine = pyocr.get_available_tools()[0]
        self.vid = None
        self.reset_video_source()
        self.history_manager = HistoryManager()
        self.digit_pattern = r"^[0-9]+$"
        self.white_lower = np.array([120, 120, 120])
        self.white_upper = np.array([255, 255, 255])

    def reset_video_source(self):
        self.vid = cv2.VideoCapture(app_option['video_source'])
        RateDetector.default_video_setting(self.vid)

    @staticmethod
    def default_video_setting(vid):
        vid.set(cv2.CAP_PROP_FPS, app_option['capture_frame_rate'])
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, app_option['capture_width'])
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, app_option['capture_height'])

    def recognize_digits(self, image):
        txt = self.ocr_engine.image_to_string(image, lang="eng",
                                              builder=pyocr.builders.DigitBuilder(tesseract_layout=7))
        if re.match(self.digit_pattern, txt):
            return int(txt)
        return 0

    def recognize_text(self, image):
        txt = self.ocr_engine.image_to_string(image, lang=app_option['ocr_language'],
                                              builder=pyocr.builders.TextBuilder(tesseract_layout=7))
        return txt.replace(' ', '')

    def crop_white_character(self, image, area):
        cropped = image[area['y']:area['y'] + area['h'], area['x']:area['x'] + area['w']]
        masked_image = cv2.inRange(cropped, self.white_lower, self.white_upper)
        masked_image = cv2.bitwise_not(masked_image)
        return Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_GRAY2RGB))

    def check_rate(self):
        _, frame = self.vid.read()
        opening, ending = False, False
        if np.shape(frame) == (app_option['capture_height'], app_option['capture_width'], 3):
            opening = self.check_opening_rate(frame)
            ending = self.check_ending_rate(frame)
        return opening, ending

    def check_opening_rate(self, frame):
        left_rate_image = self.crop_white_character(frame, app_option['left_rate_area'])
        left_rate = self.recognize_digits(left_rate_image)
        if not self.history_manager.check_rate_range(left_rate):
            return

        right_rate_image = self.crop_white_character(frame, app_option['right_rate_area'])
        right_rate = self.recognize_digits(right_rate_image)
        if not self.history_manager.check_rate_range(right_rate):
            return

        left_image = self.crop_white_character(frame, app_option['left_name_area'])
        left_name = self.recognize_text(left_image)
        right_image = self.crop_white_character(frame, app_option['right_name_area'])
        right_name = self.recognize_text(right_image)

        if left_name in app_option['user_name']:
            my_new_rate = left_rate
            opponent_new_rate = right_rate
            my_name = left_name
            opponent_name = right_name
        else:
            my_new_rate = right_rate
            opponent_new_rate = left_rate
            my_name = right_name
            opponent_name = left_name

        logging.info("%s: %s, %s: %s", my_name, my_new_rate, opponent_name, opponent_new_rate)
        return self.history_manager.record_new_rate(my_new_rate, opponent_new_rate, opponent_name)

    def check_ending_rate(self, frame):
        cropped = self.crop_white_character(frame, app_option['end_rate_area'])
        new_rate = self.recognize_digits(cropped)
        return self.history_manager.update_wl(new_rate)


class HistoryManager:
    def __init__(self):
        self.win_cnt = 0
        self.lose_cnt = 0
        self.my_rate = 0
        self.opponent_rate = 0
        self.game_history = None
        self.load_history_file()

    def load_history_file(self):
        if os.path.exists(app_option['record_file']):
            all_game_history = pd.read_csv(app_option['record_file'], sep='\t')
            self.game_history = all_game_history[-app_option["chart_period"]:]
        else:
            self.game_history = pd.DataFrame(columns=['timestamp', 'my_rate', 'opponent_rate', 'opponent_name'])
            self.game_history.to_csv(app_option['record_file'], sep='\t', index=False)
        self.game_history['my_rate'].astype('int')
        self.game_history['opponent_rate'].astype('int')

    def check_update_rate(self, before, after):
        return before != after and self.check_rate_range(after)

    def check_rate_range(self, rate):
        return app_option['lower_rate'] <= rate <= app_option['upper_rate']

    def record_new_rate(self, my_new_rate, opponent_new_rate, opponent_name):
        if (not self.check_update_rate(self.my_rate, my_new_rate)) and \
                (not self.check_update_rate(self.opponent_rate, opponent_new_rate)):
            return False

        timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.game_history = self.game_history.append(
            dict(timestamp=timestamp, my_rate=my_new_rate,
                 opponent_rate=opponent_new_rate, opponent_name=opponent_name),
            ignore_index=True)
        self.game_history.iloc[-1:].to_csv(app_option['record_file'],
                                           mode='a', index=False, header=False, sep='\t')
        self.my_rate = my_new_rate
        self.opponent_rate = opponent_new_rate
        return True

    def update_wl(self, my_new_rate):
        if not self.check_update_rate(self.my_rate, my_new_rate):
            return False
        if my_new_rate > self.my_rate:
            self.win_cnt += 1
        else:
            self.lose_cnt += 1
        self.my_rate = my_new_rate
        self.opponent_rate = 0
        return True


if __name__ == '__main__':
    main()
