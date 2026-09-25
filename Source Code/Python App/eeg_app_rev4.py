import sys
import time
import csv
import struct
import serial
import serial.tools.list_ports
import numpy as np
import pyqtgraph as pg
from pyqtgraph import InfiniteLine
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QFrame, QCheckBox, QScrollArea,
    QToolBar, QColorDialog, QFileDialog, QGridLayout, QSlider,
    QDoubleSpinBox, QSizePolicy, QDesktopWidget, QSpinBox,
    QDialog, QDialogButtonBox, QGroupBox, QRadioButton, QButtonGroup,
    QLineEdit, QProgressBar, QStackedWidget
)
from PyQt5.QtCore import QTimer, Qt, QMimeData, QVariantAnimation, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QColor, QDrag, QPixmap

# ── Signal constants ──────────────────────────────────────────────────────────
FFT_LENGTH   = 256
DISPLAY_BINS = FFT_LENGTH // 2
RAW_WINDOW_BINS = DISPLAY_BINS * 4
NUM_CHANNELS = 32
SAMPLE_RATE  = 256          # Hz — change to match your STM32 config
ADC_REFERENCE_V = 3.3
ADS1299_RAW_UNIT = "µV"
RAW_SAMPLES_PER_FRAME = max(1, round(SAMPLE_RATE * 0.020))

# UI timing: serial acquisition stays responsive, plotting is deliberately capped
# to a stable display rate so Qt/pyqtgraph is not forced to repaint every few ms.
SERIAL_POLL_MS   = 2
PLOT_INTERVAL_MS = 16   # ~62.5 FPS
PEAK_UPDATE_EVERY = 10  # peak markers/UI metadata are cheaper at ~6 Hz

# ── Binary frame protocol (must match firmware) ───────────────────────────────
PROTO_SYNC0      = 0xAA
PROTO_SYNC1      = 0x55
PROTO_HDR_SIZE   = 7                          # sync(2)+seq(2)+nch(1)+nbins(2)
PROTO_DATA_BYTES = NUM_CHANNELS * DISPLAY_BINS * 2   # uint16 magnitudes
PROTO_FRAME_SIZE = PROTO_HDR_SIZE + PROTO_DATA_BYTES + 2   # +CRC16
PROTO_SCALE      = 1.0 / 65535.0             # normalise uint16 → 0..1 float

# CRC-16/CCITT lookup table. The serial stream carries a large frame every
# 20 ms, so table lookup avoids doing eight Python bit loops per payload byte.
_crc_table = []
for _byte in range(256):
    _crc = _byte << 8
    for _ in range(8):
        _crc = ((_crc << 1) ^ 0x1021) if _crc & 0x8000 else (_crc << 1)
        _crc &= 0xFFFF
    _crc_table.append(_crc)
CRC16_TABLE = tuple(_crc_table)

# Frequency axis: bin i  →  i * (SAMPLE_RATE / FFT_LENGTH)  Hz
BIN_TO_HZ = SAMPLE_RATE / FFT_LENGTH
HZ_AXIS   = np.arange(DISPLAY_BINS, dtype=np.float32) * BIN_TO_HZ

# EEG band definitions  (name, lo_hz, hi_hz, colour)
EEG_BANDS = [
    ("δ Delta",  0.5,  4.0,  "#569CD6"),
    ("θ Theta",  4.0,  8.0,  "#4EC9B0"),
    ("α Alpha",  8.0, 13.0,  "#00B37E"),
    ("β Beta",  13.0, 30.0,  "#DCDCAA"),
    ("γ Gamma", 30.0, SAMPLE_RATE / 2, "#F48771"),
]

TRACE_PALETTE = [
    "#569CD6", "#4EC9B0", "#DCDCAA", "#F48771",
    "#9CDCFE", "#CE9178", "#B5CEA8", "#D7BA7D",
]

EEG_10_20_LABELS = [
    "Fp1", "Fpz", "Fp2", "F7",  "F3",  "Fz",  "F4",  "F8",
    "FT7", "FC3", "FCz", "FC4", "FT8", "T7",  "C3",  "Cz",
    "C4",  "T8",  "TP7", "CP3", "CPz", "CP4", "TP8", "P7",
    "P3",  "Pz",  "P4",  "P8",  "PO7", "O1",  "Oz",  "O2",
]

BAUD_RATES = ["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"]

# ── Flash LUT ─────────────────────────────────────────────────────────────────
_FLASH_LUT = []
for _i in range(256):
    _p = _i / 255.0
    _br = int(0x00 + (0x2d - 0x00) * _p); _bg = int(0xB3 + (0x2d - 0xB3) * _p); _bb = int(0x7E + (0x30 - 0x7E) * _p)
    _gr = int(0x0A + (0x1c - 0x0A) * _p); _gg = int(0x2F + (0x1c - 0x2F) * _p); _gb = int(0x24 + (0x1f - 0x24) * _p)
    _FLASH_LUT.append((f"#{_br:02x}{_bg:02x}{_bb:02x}", f"#{_gr:02x}{_gg:02x}{_gb:02x}"))



# ── Helpers ───────────────────────────────────────────────────────────────────
def make_separator():
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet("color: #2b2b2b; background-color: #2b2b2b; max-height: 1px;")
    return sep

def section_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "font-weight: bold; color: #b8d8d5; font-size: 11px; letter-spacing: 1.2px;"
        " padding: 5px 7px; background:#26363b; border-left:3px solid #46b8a8;"
        " border-radius:2px;"
    )
    return lbl

def add_band_lines(plot_widget, show=True):
    """Add labeled EEG bands and return all items so visibility can be controlled."""
    items = []
    y_top = plot_widget.viewRange()[1][1] * 0.96
    for band_index, (name, lo, hi, color) in enumerate(EEG_BANDS):
        lo_bin = lo / BIN_TO_HZ
        hi_bin = hi / BIN_TO_HZ
        band_color = QColor(color)
        band_brush = QColor(band_color)
        band_brush.setAlpha(28)
        band_pen = QColor(band_color)
        band_pen.setAlpha(170)
        region = pg.LinearRegionItem(
            values=(lo_bin, hi_bin),
            brush=pg.mkBrush(band_brush),
            movable=False,
            pen=pg.mkPen(band_pen, width=1, style=Qt.DotLine),
        )
        region.setZValue(-10)
        region.setVisible(show)
        region._eeg_band_index = band_index
        plot_widget.addItem(region)
        items.append(region)

        label = pg.TextItem(
            text=name.split()[-1].upper(),
            color=QColor(color).lighter(125),
            anchor=(0.5, 1.0),
        )
        label.setPos((lo_bin + hi_bin) / 2, y_top)
        label.setZValue(-5)
        label.setVisible(show)
        label._eeg_band_index = band_index
        plot_widget.addItem(label)
        items.append(label)
    return items


# ── Draggable plot panel ───────────────────────────────────────────────────────
class DraggablePlotWrapper(QFrame):
    def __init__(self, channel_idx, plot_widget, initial_height=180, parent=None, channel_label=None):
        super().__init__(parent)
        self.channel_idx = channel_idx
        self.plot_widget  = plot_widget

        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("PlotWrapper")
        self._base_style = "QFrame#PlotWrapper {{ background-color: {bg}; border: 1px solid {bd}; border-radius: 3px; }}"
        self.reset_style("#3e3e42", "#1e1e1e")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        ch_label = channel_label if channel_label is not None else EEG_10_20_LABELS[channel_idx]

        title_row = QHBoxLayout()
        self.title_bar = QLabel(f" ☰  {ch_label}  (CH {channel_idx+1:02d})")
        self.title_bar.setStyleSheet("""
            QLabel { background-color: #252526; color: #ffffff; font-weight: bold;
                     font-size: 12px; padding: 5px; border-radius: 2px; }
        """)
        self.title_bar.setCursor(Qt.OpenHandCursor)
        title_row.addWidget(self.title_bar, stretch=1)

        self.peak_label = QLabel("peak: –")
        self.peak_label.setStyleSheet(
            "color: #4EC9B0; font-family: 'Cascadia Code', Consolas; font-size: 11px; font-weight: bold; padding: 5px 6px;"
            " background-color: #252526; border-radius: 2px;"
        )
        title_row.addWidget(self.peak_label)

        self.quality_label = QLabel("NO DATA")
        self.quality_label.setStyleSheet(
            "color: #cccccc; font-family: Consolas; font-size: 10px; font-weight: bold; padding: 5px 6px;"
            " background-color: #252526; border-radius: 2px;"
        )
        title_row.addWidget(self.quality_label)
        layout.addLayout(title_row)
        layout.addWidget(self.plot_widget)

        self.setFixedHeight(initial_height)

        self.slide_anim = QPropertyAnimation(self, b"pos")
        self.slide_anim.setDuration(250)
        self.slide_anim.setEasingCurve(QEasingCurve.OutCubic)

        self.flash_anim = QVariantAnimation(self)
        self.flash_anim.setDuration(400)
        self.flash_anim.setStartValue(0.0)
        self.flash_anim.setEndValue(1.0)
        self.flash_anim.valueChanged.connect(self._on_flash)

    def reset_style(self, border_hex, bg_hex):
        self.setStyleSheet(self._base_style.format(bg=bg_hex, bd=border_hex))

    def _on_flash(self, progress):
        idx = min(int(progress * 255), 255)
        self.reset_style(*_FLASH_LUT[idx])

    def update_peak(self, data_row, display_mode="FFT"):
        peak_bin = int(np.argmax(data_row))
        if display_mode == "Raw Data":
            self.peak_label.setText(f"bin: {peak_bin:03d}")
        else:
            peak_hz = peak_bin * BIN_TO_HZ
            self.peak_label.setText(f"peak: {peak_hz:.1f} Hz")

    def update_quality(self, data_row):
        peak = float(np.max(data_row))
        average = float(np.mean(data_row))
        if peak <= 0.001:
            quality, color = "NO DATA", "#858585"
        elif average > 0 and peak / average > 80:
            quality, color = "CHECK", "#FFB800"
        else:
            quality, color = "OK", "#00B37E"
        self.quality_label.setText(quality)
        self.quality_label.setStyleSheet(
            f"color: {color}; font-family: Consolas; font-size: 9px; font-weight: bold; padding: 5px 6px;"
            " background-color: #252526; border-radius: 2px;"
        )

    def animate_to_pos(self, target_pos):
        self.slide_anim.stop()
        self.slide_anim.setStartValue(self.pos())
        self.slide_anim.setEndValue(target_pos)
        self.slide_anim.start()

    def trigger_drop_flash(self):
        self.flash_anim.stop()
        self.flash_anim.start()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.title_bar.geometry().contains(event.pos()):
            self.title_bar.setCursor(Qt.ClosedHandCursor)
            window = self.window()
            if hasattr(window, 'set_active_drag_origin'):
                window.set_active_drag_origin(self.channel_idx)
            drag      = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(str(self.channel_idx))
            drag.setMimeData(mime_data)
            pixmap = QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)
            drag.setHotSpot(event.pos())
            self.setVisible(False)
            drag.exec_(Qt.MoveAction)
            self.setVisible(True)
            self.title_bar.setCursor(Qt.OpenHandCursor)


# ── Grid layout ────────────────────────────────────────────────────────────────
class RearrangeableGridLayout(QGridLayout):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent

    def handle_drag_enter(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def handle_drag_move(self, event):
        if not event.mimeData().hasText():
            return
        source_idx = int(event.mimeData().text())
        target_idx = None
        for i in range(self.count()):
            w = self.itemAt(i).widget()
            if w and w.geometry().contains(event.pos()) and isinstance(w, DraggablePlotWrapper):
                if w.channel_idx != source_idx:
                    target_idx = w.channel_idx
                    break
        if target_idx is not None:
            win = self.parent_widget.window()
            if hasattr(win, 'live_swap_channels'):
                win.live_swap_channels(source_idx, target_idx)
                event.acceptProposedAction()

    def handle_drop(self, event):
        source_idx = int(event.mimeData().text())
        win = self.parent_widget.window()
        if hasattr(win, 'finalize_drop_execution'):
            win.finalize_drop_execution(source_idx)
        event.acceptProposedAction()




# ── Connection dialog ──────────────────────────────────────────────────────────
class ConnectionDialog(QDialog):
    """
    Modal dialog that handles port scanning, selection, baud rate,
    and connection status in one place.
    """
    def __init__(self, current_ser, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to Hardware")
        self.setMinimumWidth(420)
        self.setModal(True)
        self._ser = current_ser   # may be None or already open
        self.result_ser = current_ser

        self.setStyleSheet("""
            QDialog      { background-color: #1e1e1e; color: #cccccc; }
            QLabel       { color: #cccccc; font-size: 13px; }
            QGroupBox    { color: #858585; font-size: 10px; font-weight: bold;
                           border: 1px solid #3e3e42; border-radius: 3px;
                           margin-top: 8px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QComboBox    { background-color: #252526; color: #cccccc;
                           border: 1px solid #3e3e42; border-radius: 3px; padding: 5px; }
            QPushButton  { background-color: #252526; color: #cccccc; font-weight: 500;
                           border: 1px solid #3e3e42; border-radius: 3px; padding: 6px 14px; }
            QPushButton:hover { background-color: #293a40; border-color: #46b8a8; }
            QPushButton:pressed { background-color: #164a49; }
            QLabel#status_lbl { font-family: Consolas; font-size: 12px; padding: 6px;
                                 border-radius: 4px; background-color: #18181B; }
        """)

        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Port group ────────────────────────────────────────────────────
        port_group = QGroupBox("SERIAL PORT")
        port_lay   = QVBoxLayout(port_group)

        scan_row = QHBoxLayout()
        self._port_combo = QComboBox()
        self._port_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        scan_row.addWidget(self._port_combo, stretch=1)
        self._scan_btn = QPushButton("🔍 Scan")
        self._scan_btn.setFixedWidth(90)
        self._scan_btn.clicked.connect(self._scan_ports)
        scan_row.addWidget(self._scan_btn)
        port_lay.addLayout(scan_row)

        self._desc_label = QLabel("")
        self._desc_label.setStyleSheet("color:#7C7C8A; font-size:11px; padding: 2px 0;")
        port_lay.addWidget(self._desc_label)
        self._port_combo.currentIndexChanged.connect(self._on_port_selected)

        root.addWidget(port_group)

        # ── Baud rate group ───────────────────────────────────────────────
        baud_group = QGroupBox("BAUD RATE")
        baud_lay   = QVBoxLayout(baud_group)

        self._baud_combo = QComboBox()
        for b in BAUD_RATES:
            self._baud_combo.addItem(b)
        self._baud_combo.setCurrentText("921600")
        baud_lay.addWidget(self._baud_combo)

        hint = QLabel("Tip: firmware uses 921600 — match this exactly.")
        hint.setStyleSheet("color:#7C7C8A; font-size:11px;")
        baud_lay.addWidget(hint)
        root.addWidget(baud_group)

        # ── Info group ────────────────────────────────────────────────────
        info_group = QGroupBox("PROTOCOL INFO")
        info_lay   = QGridLayout(info_group)
        info_lay.setSpacing(6)

        def info_pair(row, key, val):
            k = QLabel(key)
            k.setStyleSheet("color:#7C7C8A; font-size:11px;")
            v = QLabel(val)
            v.setStyleSheet("color:#C4C4CC; font-size:11px; font-family:Consolas;")
            info_lay.addWidget(k, row, 0)
            info_lay.addWidget(v, row, 1)

        info_pair(0, "Frame sync",    "0xAA 0x55")
        info_pair(1, "Channels",      f"{NUM_CHANNELS}")
        info_pair(2, "Bins / channel",f"{DISPLAY_BINS}")
        info_pair(3, "Frame size",    f"{2+2+1+2+NUM_CHANNELS*DISPLAY_BINS*2+2} bytes")
        info_pair(4, "CRC config",    "CCITT / poly 0x1021 / init 0xFFFF")
        info_pair(5, "CRC scope",     "Payload data bytes only")
        info_pair(6, "Sample rate",   f"{SAMPLE_RATE} Hz")
        root.addWidget(info_group)

        # ── Status label ──────────────────────────────────────────────────
        self._status_lbl = QLabel("Not connected.")
        self._status_lbl.setObjectName("status_lbl")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self._status_lbl)

        # ── Action buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._conn_btn = QPushButton("🔌 Connect")
        self._conn_btn.setStyleSheet(
            "QPushButton { background-color:#238f83; border:1px solid #46b8a8; color:white; font-weight:bold; }"
            "QPushButton:hover { background-color:#2eaa9b; }"
        )
        self._conn_btn.clicked.connect(self._do_connect)

        self._disc_btn = QPushButton("⏏ Disconnect")
        self._disc_btn.setStyleSheet(
            "QPushButton { background-color:#252526; border:1px solid #3e3e42; color:#cccccc; font-weight:bold; }"
            "QPushButton:hover { background-color:#293a40; border-color:#46b8a8; }"
        )
        self._disc_btn.clicked.connect(self._do_disconnect)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(self._conn_btn)
        btn_row.addWidget(self._disc_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        # Initial scan and state sync
        self._scan_ports()
        self._sync_ui_state()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _scan_ports(self):
        self._port_combo.blockSignals(True)
        self._port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self._port_combo.addItem(p.device, userData=p.description)
        self._port_combo.blockSignals(False)
        if ports:
            self._port_combo.setCurrentIndex(0)
            self._on_port_selected(0)
            self._set_status(f"{len(ports)} port(s) found. Select one and click Connect.", "#FFB800")
        else:
            self._desc_label.setText("")
            self._set_status("No serial ports found. Check USB cable and drivers.", "#E25C5C")

    def _on_port_selected(self, idx):
        desc = self._port_combo.itemData(idx)
        self._desc_label.setText(desc or "")

    def _do_connect(self):
        port = self._port_combo.currentText()
        if not port:
            self._set_status("No port selected.", "#E25C5C")
            return
        baud = int(self._baud_combo.currentText())
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
            self._ser = serial.Serial(port, baudrate=baud, timeout=0.02)
            self._ser.reset_input_buffer()
            self.result_ser = self._ser
            self._set_status(f"✓  Connected: {port} @ {baud} baud", "#00B37E")
        except Exception as e:
            self._ser = None
            self.result_ser = None
            self._set_status(f"✗  {e}", "#E25C5C")
        self._sync_ui_state()

    def _do_disconnect(self):
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
        self._ser = None
        self.result_ser = None
        self._set_status("Disconnected.", "#FFB800")
        self._sync_ui_state()

    def _set_status(self, text, color):
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(
            f"QLabel#status_lbl {{ color:{color}; font-family:Consolas; font-size:12px;"
            f" padding:6px; border-radius:4px; background-color:#18181B; }}"
        )

    def _sync_ui_state(self):
        connected = bool(self._ser and self._ser.is_open)
        self._conn_btn.setEnabled(not connected)
        self._disc_btn.setEnabled(connected)
        self._port_combo.setEnabled(not connected)
        self._baud_combo.setEnabled(not connected)
        self._scan_btn.setEnabled(not connected)

    # ── Read-back accessors ───────────────────────────────────────────────
    @property
    def serial_port(self):
        return self.result_ser

    @property
    def selected_port_name(self):
        return self._port_combo.currentText()

    @property
    def selected_baud(self):
        return int(self._baud_combo.currentText())

# ── Recording setup dialog ─────────────────────────────────────────────────────
class RecordingDialog(QDialog):
    """Modal dialog to configure a recording session before it starts."""

    def __init__(self, channel_labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Recording Session")
        self.setMinimumWidth(460)
        self.setStyleSheet("""
            QDialog        { background-color: #1e1e1e; color: #cccccc; }
            QLabel         { color: #cccccc; font-size: 13px; }
            QGroupBox      { color: #858585; font-size: 10px; font-weight: bold;
                             border: 1px solid #3e3e42; border-radius: 3px;
                             margin-top: 8px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QCheckBox      { color: #C4C4CC; font-size: 12px; padding: 2px; }
            QCheckBox:hover{ color: white; }
            QRadioButton   { color: #C4C4CC; font-size: 12px; padding: 2px; }
            QRadioButton:hover { color: white; }
            QSpinBox, QLineEdit {
                background-color: #202024; color: white;
                border: 1px solid #323238; border-radius: 4px; padding: 4px; }
            QPushButton    { background-color: #252526; color: #cccccc; font-weight: 500;
                             border: 1px solid #3e3e42; border-radius: 3px; padding: 6px 14px; }
            QPushButton:hover { background-color: #293a40; border-color: #46b8a8; }
        """)

        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Duration ──────────────────────────────────────────────────────
        dur_group = QGroupBox("DURATION")
        dur_layout = QVBoxLayout(dur_group)
        dur_layout.setSpacing(8)

        self._dur_radio_group = QButtonGroup(self)
        self._rb_unlimited = QRadioButton("Record until manually stopped")
        self._rb_timed     = QRadioButton("Stop automatically after:")
        self._rb_unlimited.setChecked(True)
        self._dur_radio_group.addButton(self._rb_unlimited)
        self._dur_radio_group.addButton(self._rb_timed)
        dur_layout.addWidget(self._rb_unlimited)

        timed_row = QHBoxLayout()
        timed_row.addWidget(self._rb_timed)
        self._dur_spin = QSpinBox()
        self._dur_spin.setRange(1, 86400)
        self._dur_spin.setValue(60)
        self._dur_spin.setFixedWidth(72)
        self._dur_spin.setEnabled(False)
        timed_row.addWidget(self._dur_spin)
        timed_row.addWidget(QLabel("seconds"))
        timed_row.addStretch()
        dur_layout.addLayout(timed_row)

        self._rb_timed.toggled.connect(self._dur_spin.setEnabled)
        root.addWidget(dur_group)

        # ── Channels ──────────────────────────────────────────────────────
        ch_group = QGroupBox("CHANNELS TO RECORD")
        ch_vlay  = QVBoxLayout(ch_group)
        ch_vlay.setSpacing(6)

        macro_row = QHBoxLayout()
        sel_all = QPushButton("Select All")
        sel_none = QPushButton("Clear All")
        sel_active = QPushButton("Active Only")
        for btn in (sel_all, sel_none, sel_active):
            btn.setFixedHeight(26)
            macro_row.addWidget(btn)
        macro_row.addStretch()
        ch_vlay.addLayout(macro_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(160)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #2d2d30; }")
        inner = QWidget()
        inner.setStyleSheet("background-color: #18181B;")
        grid  = QGridLayout(inner)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(4)

        self._ch_checks = []
        for i, lbl in enumerate(channel_labels):
            cb = QCheckBox(f"({lbl}) Ch {i+1:02d}")
            cb.setChecked(True)
            grid.addWidget(cb, i // 4, i % 4)
            self._ch_checks.append(cb)

        scroll.setWidget(inner)
        ch_vlay.addWidget(scroll)
        root.addWidget(ch_group)

        sel_all.clicked.connect(lambda: [cb.setChecked(True)  for cb in self._ch_checks])
        sel_none.clicked.connect(lambda: [cb.setChecked(False) for cb in self._ch_checks])
        # "Active Only" is wired by the caller after construction via set_active_channels()

        self._sel_active_btn = sel_active

        # ── Output file ───────────────────────────────────────────────────
        file_group = QGroupBox("OUTPUT FILE")
        file_lay   = QVBoxLayout(file_group)

        file_row = QHBoxLayout()
        self._path_edit = QLineEdit(f"eeg_{int(time.time())}.csv")
        self._path_edit.setPlaceholderText("output path…")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(self._path_edit, stretch=1)
        file_row.addWidget(browse_btn)
        file_lay.addLayout(file_row)
        root.addWidget(file_group)

        # ── Buttons ───────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("▶  Start Recording")
        btns.button(QDialogButtonBox.Ok).setStyleSheet(
            "QPushButton { background-color:#238f83; border:1px solid #46b8a8; color:white; }"
            "QPushButton:hover { background-color:#2eaa9b; }"
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Recording", self._path_edit.text(), "CSV (*.csv)"
        )
        if path:
            self._path_edit.setText(path)

    def set_active_channels(self, active_indices):
        """Pre-select only the currently visible channels when 'Active Only' is clicked."""
        self._sel_active_btn.clicked.connect(
            lambda: [cb.setChecked(i in active_indices) for i, cb in enumerate(self._ch_checks)]
        )

    # ── Result accessors ──────────────────────────────────────────────────
    @property
    def output_path(self):
        return self._path_edit.text().strip()

    @property
    def duration_seconds(self):
        """Returns None for unlimited, int for timed."""
        return self._dur_spin.value() if self._rb_timed.isChecked() else None

    @property
    def selected_channels(self):
        return [i for i, cb in enumerate(self._ch_checks) if cb.isChecked()]


class PreferencesDialog(QDialog):
    def __init__(self, accent, background, preview_enabled, theme, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EEG Studio Preferences")
        self.setMinimumWidth(430)
        accent_hover = QColor(accent).lighter(118).name()
        self.setStyleSheet(
            "QDialog { background:#151a1d; color:#d7e0e3; }"
            "QLabel { color:#b5c2c5; }"
            "QGroupBox { color:#769096; border:1px solid #35434a; border-radius:5px; margin-top:10px; padding:12px; }"
            "QComboBox, QCheckBox { color:#d7e0e3; background:#20282d; }"
            "QPushButton { background:#202c31; color:#d7e0e3; border:1px solid #405159; border-radius:4px; padding:7px 12px; }"
            f"QPushButton:hover {{ background:#293a40; border-color:{accent_hover}; }}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QLabel("PREFERENCES")
        title.setStyleSheet("color:#f1f7f6; font-size:16px; font-weight:bold; letter-spacing:1px;")
        root.addWidget(title)

        appearance = QGroupBox("APPEARANCE")
        appearance_layout = QVBoxLayout(appearance)
        accent_row = QHBoxLayout()
        accent_row.addWidget(QLabel("UI accent color"))
        self.accent_button = QPushButton("      ")
        self.accent_button.setFixedWidth(72)
        self._accent = accent
        self._refresh_accent_button()
        self.accent_button.clicked.connect(self.choose_accent)
        accent_row.addWidget(self.accent_button)
        accent_row.addStretch()
        appearance_layout.addLayout(accent_row)

        background_row = QHBoxLayout()
        background_row.addWidget(QLabel("Background color"))
        self.background_button = QPushButton("      ")
        self.background_button.setFixedWidth(72)
        self._background = background
        self._refresh_background_button()
        self.background_button.clicked.connect(self.choose_background)
        background_row.addWidget(self.background_button)
        background_row.addStretch()
        appearance_layout.addLayout(background_row)

        self.preview_check = QCheckBox("Show live preview on Home")
        self.preview_check.setChecked(preview_enabled)
        appearance_layout.addWidget(self.preview_check)
        root.addWidget(appearance)

        theme_group = QGroupBox("INTERFACE THEME")
        theme_layout = QVBoxLayout(theme_group)
        theme_layout.addWidget(QLabel("Choose the overall interface color system"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Teal Console", "Monochrome"])
        self.theme_combo.setCurrentText("Monochrome" if theme == "Monochrome" else "Teal Console")
        theme_layout.addWidget(self.theme_combo)
        root.addWidget(theme_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Apply preferences")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        defaults_button = QPushButton("Restore defaults")
        defaults_button.clicked.connect(self.restore_defaults)
        buttons.addButton(defaults_button, QDialogButtonBox.ResetRole)
        root.addWidget(buttons)

    def _refresh_accent_button(self):
        self.accent_button.setStyleSheet(
            f"QPushButton {{ background:{self._accent}; border:1px solid #f1f7f6; border-radius:4px; }}"
        )

    def choose_accent(self):
        color = QColorDialog.getColor(QColor(self._accent), self, "Choose UI Accent")
        if color.isValid():
            self._accent = color.name()
            self._refresh_accent_button()

    def _refresh_background_button(self):
        self.background_button.setStyleSheet(
            f"QPushButton {{ background:{self._background}; border:1px solid #f1f7f6; border-radius:4px; }}"
        )

    def choose_background(self):
        color = QColorDialog.getColor(QColor(self._background), self, "Choose Background Color")
        if color.isValid():
            self._background = color.name()
            self._refresh_background_button()

    def restore_defaults(self):
        self._accent = "#46b8a8"
        self._background = "#111416"
        self.preview_check.setChecked(True)
        self.theme_combo.setCurrentText("Teal Console")
        self._refresh_accent_button()
        self._refresh_background_button()

    @property
    def selected_accent(self):
        return self._accent

    @property
    def selected_background(self):
        return self._background


# ── Main application ───────────────────────────────────────────────────────────
class MultiChannelFFTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("32-Channel EEG Signal Analyzer")

        screen = QDesktopWidget().availableGeometry()
        dw = int(screen.width()  * 0.85)
        dh = int(screen.height() * 0.85)
        self.resize(dw, dh)
        self.setAcceptDrops(True)

        self.setStyleSheet("""
            QMainWindow  { background-color: #111416; }
            QToolBar     { background-color: #171c20; border-bottom: 1px solid #334047; padding: 4px 8px; }
            QLabel       { color: #cccccc; font-family: 'Segoe UI', Arial; font-size: 13px; }
            QComboBox    { background-color: #252526; color: #cccccc; border: 1px solid #3e3e42;
                           border-radius: 3px; padding: 5px 8px; min-width: 80px; }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #46b8a8; }
            QComboBox QAbstractItemView { background-color: #252526; color: #cccccc; selection-background-color: #164a49; }
            QPushButton  { background-color: #20282d; color: #d7e0e3; font-weight: 500;
                           border: 1px solid #3b4a51; border-radius: 4px; padding: 7px 12px; }
            QPushButton:hover { background-color: #29363d; border-color: #46b8a8; }
            QPushButton:pressed { background-color: #164a49; }
            QScrollArea  { border: none; background-color: #111416; }
            QCheckBox    { color: #cccccc; font-family: 'Segoe UI'; font-size: 12px; padding: 3px; }
            QCheckBox:hover { color: white; }
            QDoubleSpinBox, QSpinBox {
                background-color: #252526; color: white; border: 1px solid #3e3e42;
                border-radius: 3px; padding: 4px; min-width: 65px; }
            QSlider::groove:horizontal { border:1px solid #3e3e42; height:6px; background:#252526; border-radius:3px; }
            QSlider::handle:horizontal { background:#238f83; width:14px; margin:-4px 0; border-radius:7px; }
            QSlider::handle:horizontal:hover { background:#46b8a8; }
            QScrollBar:vertical { background:#181818; width:10px; margin:0; }
            QScrollBar::handle:vertical { background:#424242; min-height:24px; border-radius:3px; }
            QScrollBar::handle:vertical:hover { background:#5a5a5a; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
            QFrame#HomePage { background-color:#151a1d; border:1px solid #334047; border-radius:5px; }
            QFrame#HomeHero { background-color:#1b252a; border:1px solid #3b4a51; border-left:4px solid #46b8a8; border-radius:5px; }
            QFrame#MetricCard { background-color:#1d2529; border:1px solid #35434a; border-radius:5px; }
            QFrame#MetricCard:hover { background-color:#222e33; border-color:#46b8a8; }
            QFrame#HomeActions { background-color:#1b252a; border:1px solid #35434a; border-radius:5px; }
            QLabel#HomeEyebrow { color:#6f9b98; font-size:10px; font-weight:bold; letter-spacing:1px; }
            QLabel#ToolbarGroup { color:#6f9b98; font-family:'Consolas'; font-size:9px; font-weight:bold; letter-spacing:0.8px; padding-left:4px; }
            QLabel#WorkspaceBadge { color:#d7e0e3; background:#202c31; border:1px solid #405159; border-radius:4px; padding:5px 9px; font-family:'Consolas'; font-size:10px; font-weight:bold; letter-spacing:0.6px; }
            QLabel#TelemetryBadge { color:#68d6bd; background:#173b3a; border:1px solid #28685f; border-radius:4px; padding:4px 8px; font-family:'Consolas'; font-size:10px; font-weight:bold; }
            QLabel#HomeTitle { color:#f1f7f6; font-size:27px; font-weight:bold; }
            QLabel#FounderSignature { color:#d9b36c; font-family:'Segoe Script','Brush Script MT',cursive; font-size:25px; font-style:italic; padding-left:8px; }
            QLabel#HomeHeroDetail { color:#9aadb1; font-size:12px; }
            QLabel#SignalProfile { color:#68d6bd; font-family:'Consolas'; font-size:10px; font-weight:bold; letter-spacing:0.5px; padding-top:5px; }
            QLabel#PreviewMode { color:#68d6bd; font-family:'Consolas'; font-size:10px; font-weight:bold; }
            QLabel#HomeEventTicker { color:#9aadb1; background:#1b252a; border:1px solid #35434a; border-left:3px solid #d9b36c; border-radius:4px; padding:9px 12px; font-family:'Consolas'; font-size:10px; }
            QLabel#HomeStatus { color:#68d6bd; background:#173b3a; border:1px solid #28685f; border-radius:4px; padding:7px 11px; font-size:10px; font-weight:bold; }
            QPushButton#HomeConnectAction { background:#238f83; color:#ffffff; border:1px solid #68d6bd; border-radius:4px; padding:7px 14px; font-weight:bold; }
            QPushButton#HomeConnectAction:hover { background:#2eaa9b; }
            QLabel#MetricValue { color:#f1f7f6; font-size:25px; font-weight:bold; }
            QLabel#MetricLabel { color:#769096; font-size:10px; font-weight:bold; letter-spacing:0.8px; }
            QLabel#MetricDetail { color:#b5c2c5; font-size:11px; }
            QPushButton#PrimaryAction { background-color:#238f83; color:white; border:1px solid #46b8a8; font-weight:bold; padding:10px 16px; }
            QPushButton#PrimaryAction:hover { background-color:#2eaa9b; }
            QPushButton#SecondaryAction { background-color:#202c31; color:#d7e0e3; border:1px solid #405159; padding:10px 16px; }
            QPushButton#SecondaryAction:hover { background-color:#293a40; border-color:#46b8a8; }
            QPushButton#HardwareAction { background-color:#173b3a; color:#68d6bd; border:1px solid #28685f; padding:10px 16px; font-weight:bold; }
            QPushButton#HardwareAction:hover { background-color:#20524e; border-color:#68d6bd; }
        """)

        self._base_app_style = self.styleSheet()

        # ── State ──────────────────────────────────────────────────────────────
        self.ser              = None
        self.is_split_view    = False
        self.is_frozen        = False
        self.show_bands       = True
        self.split_columns    = 2
        self.current_plot_height = int(dh * 0.22)
        self.calibration_scalar  = 1.0
        self.packet_count     = 0
        self.dropped_packets  = 0
        self._last_packet_count = 0
        self._last_throughput_time = time.time()
        self._throughput_bps = 0.0
        self.last_fps_time    = time.time()
        self.session_runtime  = 0.0
        self.last_update_time = time.time()
        self.active_drag_origin_idx = None
        self.is_animating_layout    = False
        self.grid_layout      = None
        self.nav_visible      = True
        self._recording       = False
        self._csv_writer      = None
        self._csv_file        = None
        self._peak_tick       = 0      # update peak labels every N frames
        self._serial_buf      = bytearray()   # raw bytes from UART
        self._last_seq        = -1            # for out-of-order detection
        self._last_sent_stream_command = None  # last STM32 mode byte actually sent
        self._crc_errors      = 0
        self._record_duration = None   # None = unlimited
        self._record_channels = list(range(NUM_CHANNELS))
        self._record_elapsed  = 0.0
        self._record_timer    = None
        self.display_mode     = "FFT"
        # UI workspace selection is separate from the display renderer so the
        # STM32 can be told exactly which acquisition/view mode is active.
        self.workspace_mode   = "FFT"
        self.custom_sensors_mode = False   # True while the "Custom Sensors" workspace is active
        self.active_sensor_count = 4       # kept in sync with the sidebar checkboxes
        self.show_peak_markers = True
        self.peak_lines        = {}
        self.peak_labels       = {}
        self.plot_signal_proxies = []
        self.cursor_readout    = None
        self.auto_scale_y      = False
        self._frame_ready      = False
        self.ui_accent         = "#46b8a8"
        self.ui_background     = "#111416"
        self.ui_theme          = "Teal Console"
        self.home_preview_enabled = True

        self.channel_render_order = list(range(NUM_CHANNELS))
        self.channel_colors = [
            QColor(TRACE_PALETTE[i % len(TRACE_PALETTE)]).lighter(
                100 + 12 * (i // len(TRACE_PALETTE))
            )
            for i in range(NUM_CHANNELS)
        ]

        self.fft_data_matrix = np.zeros((NUM_CHANNELS, DISPLAY_BINS), dtype=np.float32)

        # Raw acquisition uses a ring buffer. Samples are appended in-place and
        # the GUI materializes one chronological view only when new data arrives.
        self.raw_data_matrix = np.zeros((NUM_CHANNELS, RAW_WINDOW_BINS), dtype=np.float32)
        # Oscilloscope-style time axis: sample index -> seconds
        self.raw_x_axis = np.arange(RAW_WINDOW_BINS, dtype=np.float32) / SAMPLE_RATE
        self._raw_ring_write = 0
        self._raw_ring_count = 0
        self._raw_new_samples = 0
        self._raw_render_matrix = np.zeros_like(self.raw_data_matrix)
        self._raw_indices = np.arange(RAW_WINDOW_BINS, dtype=np.int32)

        # ── Build UI ──────────────────────────────────────────────────────────
        self.init_control_toolbar()

        main_workspace = QWidget()
        self.setCentralWidget(main_workspace)
        self.layout_core = QHBoxLayout(main_workspace)
        self.layout_core.setContentsMargins(12, 12, 12, 12)
        self.layout_core.setSpacing(12)

        self.content_stack = QStackedWidget()
        self.layout_core.addWidget(self.content_stack, stretch=1)

        self.plot_container = QWidget()
        self.plot_container.setObjectName("PlotWorkspace")
        self.plot_container.setStyleSheet(
            "QWidget#PlotWorkspace { background-color: #1e1e1e; border: 1px solid #2b2b2b; border-radius: 3px; }"
        )
        self.plot_container_layout = QVBoxLayout(self.plot_container)
        self.plot_container_layout.setContentsMargins(0, 0, 0, 0)
        self.content_stack.addWidget(self.plot_container)

        self.init_plot_display()

        self.curves           = {}
        self.individual_widgets = {}
        self.color_buttons    = []
        self.band_region_items = []   # for merged view

        self.init_left_navigation_menu(self.layout_core)
        self.init_home_dashboard()
        self.show_home()
        self.update_channel_visibility()

        self.serial_timer = QTimer()
        self.serial_timer.timeout.connect(self._poll_serial)
        self.serial_timer.setTimerType(Qt.PreciseTimer)
        self.serial_timer.start(SERIAL_POLL_MS)

        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.update_plots)
        self.render_timer.setTimerType(Qt.PreciseTimer)
        self.render_timer.start(PLOT_INTERVAL_MS)

    # ══════════════════════════════════════════════════════════════════════════
    # Home dashboard
    # ══════════════════════════════════════════════════════════════════════════
    def init_home_dashboard(self):
        self.home_page = QFrame()
        self.home_page.setObjectName("HomePage")

        root = QVBoxLayout(self.home_page)
        root.setContentsMargins(28, 26, 28, 26)
        root.setSpacing(16)

        hero = QFrame()
        hero.setObjectName("HomeHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(16, 12, 16, 12)
        hero_layout.setSpacing(4)

        eyebrow = QLabel("EEG STUDIO  /  LIVE SIGNAL CONSOLE")
        eyebrow.setObjectName("HomeEyebrow")
        hero_layout.addWidget(eyebrow)

        hero_row = QHBoxLayout()
        title = QLabel("Signal control room")
        title.setObjectName("HomeTitle")
        hero_row.addWidget(title)
        signature = QLabel("FS")
        signature.setObjectName("FounderSignature")
        signature.setToolTip("FS")
        hero_row.addWidget(signature)
        hero_row.addStretch()

        self.home_status_label = QLabel("●  SYSTEM READY")
        self.home_status_label.setObjectName("HomeStatus")
        hero_row.addWidget(self.home_status_label)

        self.home_connect_btn = QPushButton("Connect hardware")
        self.home_connect_btn.setObjectName("HomeConnectAction")
        self.home_connect_btn.setFixedHeight(34)
        self.home_connect_btn.clicked.connect(self.open_connection_dialog)
        hero_row.addWidget(self.home_connect_btn)
        hero_layout.addLayout(hero_row)

        subtitle = QLabel(
            "Connect the device, verify the stream, then move into the live analyzer."
        )
        subtitle.setObjectName("HomeHeroDetail")
        hero_layout.addWidget(subtitle)

        signal_profile = QLabel(
            f"◌  {NUM_CHANNELS} CHANNELS   /   {SAMPLE_RATE} HZ   /   USB CDC   /   CRC-16"
        )
        signal_profile.setObjectName("SignalProfile")
        hero_layout.addWidget(signal_profile)
        root.addWidget(hero)

        snapshot_label = QLabel("SESSION SNAPSHOT")
        snapshot_label.setObjectName("MetricLabel")
        root.addWidget(snapshot_label)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        cards.setVerticalSpacing(12)

        def add_metric(row, col, label, value, detail, accent):
            card = QFrame()
            card.setObjectName("MetricCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 12, 14, 12)
            card_layout.setSpacing(4)

            metric_label = QLabel(label)
            metric_label.setObjectName("MetricLabel")
            card_layout.addWidget(metric_label)

            value_label = QLabel(value)
            value_label.setObjectName("MetricValue")
            value_label.setStyleSheet(f"color:{accent}; font-size:22px; font-weight:bold;")
            card_layout.addWidget(value_label)

            detail_label = QLabel(detail)
            detail_label.setObjectName("MetricDetail")
            card_layout.addWidget(detail_label)
            card.setMinimumHeight(92)
            cards.addWidget(card, row, col)
            return value_label, detail_label

        self.home_connection_value, self.home_connection_detail = add_metric(
            0, 0, "HARDWARE LINK", "OFFLINE", "Scan a port to begin", "#f48771"
        )
        self.home_channels_value, self.home_channels_detail = add_metric(
            0, 1, "ACTIVE CHANNELS", "04 / 32", "Visible in the analyzer", "#4ec9b0"
        )
        self.home_packets_value, self.home_packets_detail = add_metric(
            1, 0, "PACKETS RECEIVED", "0", "CRC errors: 0", "#569cd6"
        )
        self.home_runtime_value, self.home_runtime_detail = add_metric(
            1, 1, "SESSION RUNTIME", "00:00:00", "Ready for acquisition", "#dcdcaa"
        )
        self.home_throughput_value, self.home_throughput_detail = add_metric(
            2, 0, "THROUGHPUT", "0.0 kbps", "Waiting for stream", "#9cdcfe"
        )
        self.home_health_value, self.home_health_detail = add_metric(
            2, 1, "STREAM HEALTH", "IDLE", "No active acquisition", "#dcdcaa"
        )
        root.addLayout(cards)

        preview_panel = QFrame()
        preview_panel.setObjectName("HomeActions")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(14, 12, 14, 12)
        preview_layout.setSpacing(8)
        preview_header = QHBoxLayout()
        preview_title = QLabel("LIVE SIGNAL PREVIEW")
        preview_title.setObjectName("MetricLabel")
        preview_header.addWidget(preview_title)
        preview_header.addStretch()
        self.home_preview_mode = QLabel("FFT / CHANNEL 01")
        self.home_preview_mode.setObjectName("PreviewMode")
        preview_header.addWidget(self.home_preview_mode)
        preview_layout.addLayout(preview_header)

        self.home_preview = pg.PlotWidget()
        self.home_preview.setFixedHeight(130)
        self.home_preview.setBackground('#12191c')
        self.home_preview.hideAxis('left')
        self.home_preview.hideAxis('bottom')
        self.home_preview.setMouseEnabled(x=False, y=False)
        self.home_preview.setMenuEnabled(False)
        self.home_preview.showGrid(x=False, y=False)
        self.home_preview_curve = self.home_preview.plot(
            x=HZ_AXIS,
            y=np.zeros(DISPLAY_BINS, dtype=np.float32),
            pen=pg.mkPen('#46b8a8', width=1.5),
        )
        preview_layout.addWidget(self.home_preview)
        root.addWidget(preview_panel)

        link_panel = QFrame()
        link_panel.setObjectName("HomeActions")
        link_layout = QVBoxLayout(link_panel)
        link_layout.setContentsMargins(14, 12, 14, 12)
        link_layout.setSpacing(8)

        link_title = QLabel("SERIAL LINK PROFILE")
        link_title.setObjectName("MetricLabel")
        link_layout.addWidget(link_title)

        link_grid = QGridLayout()
        link_grid.setHorizontalSpacing(18)
        link_grid.setVerticalSpacing(4)
        link_values = [
            ("Frame sync", "0xAA  0x55"),
            ("Payload", f"{NUM_CHANNELS} channels  /  {DISPLAY_BINS} bins"),
            ("CRC config", "CRC-16/CCITT  |  poly 0x1021  |  init 0xFFFF"),
            ("CRC scope", "Payload data bytes only"),
            ("Sample rate", f"{SAMPLE_RATE} Hz"),
        ]
        for index, (label_text, value_text) in enumerate(link_values):
            label = QLabel(label_text.upper())
            label.setObjectName("MetricLabel")
            value = QLabel(value_text)
            value.setObjectName("MetricDetail")
            link_grid.addWidget(label, index, 0)
            link_grid.addWidget(value, index, 1)
        link_grid.setColumnStretch(1, 1)
        link_layout.addLayout(link_grid)
        root.addWidget(link_panel)

        self.home_event_label = QLabel("EVENTS  /  No warnings recorded this session")
        self.home_event_label.setObjectName("HomeEventTicker")
        self.home_event_label.setWordWrap(True)
        root.addWidget(self.home_event_label)
        root.addStretch()

        self.content_stack.addWidget(self.home_page)

    def show_home(self):
        self.content_stack.setCurrentWidget(self.home_page)
        self.home_nav_btn.setChecked(True)
        self.workspace_label.setText("OVERVIEW")
        self._set_dashboard_controls_visible(False)
        self._set_custom_controls_visible(False)
        self._set_fft_controls_visible(False)
        self.nav_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.nav_frame.setMinimumHeight(0)
        self.nav_frame.setMaximumHeight(16777215)
        self.layout_core.setAlignment(self.nav_frame, Qt.AlignTop)

    def select_workspace_mode(self, mode):
        """Select one of the four acquisition/view workspaces and notify STM32.

        Existing firmware commands F/R remain unchanged for FFT/EEG Raw.
        Sensors and General Oscilloscope use the explicit S/O mode bytes; the
        STM32 firmware must implement those two mode commands.
        """
        mode = str(mode)
        if mode not in ("FFT", "EEG Raw", "Sensors", "Oscilloscope"):
            return
        self.workspace_mode = mode
        self.custom_sensors_mode = mode in ("Sensors", "Oscilloscope")
        display_mode = "FFT" if mode == "FFT" else "Raw Data"
        if self.display_mode != display_mode:
            self.set_display_mode(display_mode)
        self._refresh_workspace_ui()

    def _reset_acquisition_session(self):
        """Clear parser/session state. Needed both on a fresh connection and
        whenever we send the STM32 a mode-change command: the firmware resets
        its own frame sequence counter when it switches modes, so if we don't
        also reset _last_seq here, every frame of the new stream looks "old"
        compared to the last sequence number from the previous mode and gets
        silently dropped forever (acquisition appears frozen after a mode switch)."""
        self._serial_buf.clear()
        self._raw_ring_write = 0
        self._raw_ring_count = 0
        self._raw_new_samples = 0
        self.raw_data_matrix.fill(0.0)
        self._raw_render_matrix.fill(0.0)
        self._frame_ready = False
        self._last_seq       = -1
        self._crc_errors     = 0
        self.packet_count    = 0
        self.dropped_packets = 0

    def _send_workspace_command(self, mode=None):
        mode = mode or self.workspace_mode
        command = {
            "FFT": b"F",
            "EEG Raw": b"R",
            "Sensors": b"S",
            "Oscilloscope": b"O",
        }.get(mode)
        if command is None or not self.ser or not self.ser.is_open:
            return
        if command != self._last_sent_stream_command:
            self._reset_acquisition_session()
        if self._send_stream_command(command):
            self._last_sent_stream_command = command

    def _refresh_workspace_ui(self):
        self.content_stack.setCurrentWidget(self.plot_container)
        self.home_nav_btn.setChecked(False)
        for button in self.mode_buttons:
            button.setChecked(button.property("stream_mode") == self.workspace_mode)
        names = {
            "FFT": ("FFT WORKSPACE", "Frequency-domain EEG spectrum  |  0–128 Hz"),
            "EEG Raw": ("EEG RAW OSCILLOSCOPE", f"EEG waveform  |  {ADS1299_RAW_UNIT} vs Time (s)"),
            "Sensors": ("SENSOR OSCILLOSCOPE", f"{self.active_sensor_count} active sensor channel(s)  |  {ADS1299_RAW_UNIT} vs Time (s)"),
            "Oscilloscope": ("GENERAL OSCILLOSCOPE", f"General analog waveform  |  {ADS1299_RAW_UNIT} vs Time (s)"),
        }
        title, detail = names[self.workspace_mode]
        self.workspace_label.setText(title)
        self.workspace_mode_label.setText(self.workspace_mode.upper())
        self.graph_title_label.setText(title)
        self.graph_detail_label.setText(detail)
        self._set_dashboard_controls_visible(True)
        self._set_custom_controls_visible(self.workspace_mode == "Sensors")
        self._set_fft_controls_visible(self.workspace_mode == "FFT")
        self._refresh_channel_labels()
        self._send_workspace_command()

    def show_dashboard(self, mode=None):
        if mode is not None:
            self.select_workspace_mode({"Raw Data": "EEG Raw"}.get(mode, mode))
        else:
            self._refresh_workspace_ui()

    def show_custom_dashboard(self):
        self.select_workspace_mode("Sensors")

    def show_oscilloscope_dashboard(self):
        self.select_workspace_mode("Oscilloscope")

    def _set_custom_controls_visible(self, visible):
        for widget in getattr(self, "custom_only_widgets", []):
            widget.setVisible(visible)

    def _set_fft_controls_visible(self, visible):
        for widget in getattr(self, "fft_only_widgets", []):
            widget.setVisible(visible)

    @staticmethod
    def _set_layout_item_visible(item, visible):
        widget = item.widget()
        if widget is not None:
            widget.setVisible(visible)
            return
        child_layout = item.layout()
        if child_layout is not None:
            for child_index in range(child_layout.count()):
                MultiChannelFFTApp._set_layout_item_visible(
                    child_layout.itemAt(child_index), visible
                )

    def _set_dashboard_controls_visible(self, visible):
        for widget in self.dashboard_toolbar_widgets:
            widget.setVisible(visible)

        nav_layout = self.nav_frame.layout()
        for item_index in range(
            self.dashboard_nav_start_index,
            self.utility_nav_start_index,
        ):
            self._set_layout_item_visible(nav_layout.itemAt(item_index), visible)

    def open_preferences(self):
        dialog = PreferencesDialog(
            self.ui_accent,
            self.ui_background,
            self.home_preview_enabled,
            self.ui_theme,
            self,
        )
        if dialog.exec_() == QDialog.Accepted:
            self.ui_accent = dialog.selected_accent
            self.ui_background = dialog.selected_background
            self.ui_theme = "Monochrome" if dialog.theme_combo.currentText() == "Monochrome" else "Teal Console"
            self.home_preview_enabled = dialog.preview_check.isChecked()
            self.home_preview.setVisible(self.home_preview_enabled)
            self._apply_ui_accent()
            self._set_status("Preferences applied.", self.ui_accent)

    def _apply_ui_accent(self):
        monochrome = self.ui_theme == "Monochrome"
        accent_value = "#ffffff" if monochrome else self.ui_accent
        background_value = "#000000" if monochrome else self.ui_background
        accent = QColor(accent_value)
        accent_hover = accent.lighter(118).name()
        accent_deep = accent.darker(155).name()
        accent_dark = accent.darker(125).name()
        accent_pale = accent.lighter(135).name()

        palette_replacements = {
            "#111416": background_value,
            "#46b8a8": accent_value,
            "#68d6bd": accent_pale,
            "#238f83": accent_dark,
            "#2eaa9b": accent_hover,
            "#164a49": accent_deep,
            "#20524e": accent_dark,
            "#173b3a": accent_deep,
            "#28685f": accent_value,
        }
        if monochrome:
            palette_replacements.update({
                "#111416": "#000000",
                "#171c20": "#080808",
                "#20282d": "#121212",
                "#1b252a": "#181818",
                "#1d2529": "#1f1f1f",
                "#334047": "#383838",
                "#35434a": "#3d3d3d",
                "#3b4a51": "#484848",
                "#405159": "#505050",
                "#293a40": "#292929",
                "#202c31": "#1b1b1b",
                "#d9b36c": "#d0d0d0",
                "#6f9b98": "#a0a0a0",
                "#9aadb1": "#bdbdbd",
                "#769096": "#a8a8a8",
                "#b5c2c5": "#c8c8c8",
                "#d7e0e3": "#e5e5e5",
            })
        app_style = self._base_app_style
        for old_color, new_color in palette_replacements.items():
            app_style = app_style.replace(old_color, new_color)
        self.setStyleSheet(app_style)

        self.nav_frame.setStyleSheet(
            f"background-color:{background_value}; border-radius:5px; border:1px solid #334047;"
        )
        self.home_page.setStyleSheet(
            f"QFrame#HomePage {{ background-color:{background_value}; border:1px solid #334047; border-radius:5px; }}"
        )
        self.plot_container.setStyleSheet(
            f"QWidget#PlotWorkspace {{ background-color:{background_value}; border:1px solid #334047; border-radius:3px; }}"
        )
        self.plot_widget.setBackground(background_value)
        self.home_preview.setBackground(background_value)
        if self.is_split_view:
            for wrapper in self.individual_widgets.values():
                wrapper.plot_widget.setBackground(background_value)

        self.brand_label.setStyleSheet(
            f"color:{accent_pale}; font-size:13px; font-weight:bold; letter-spacing:0.8px; padding-right:12px;"
        )
        self.runtime_label.setStyleSheet(
            f"color:{accent_pale}; font-family:'Consolas',monospace; font-size:13px; font-weight:bold; margin-right:10px;"
        )
        self.graph_status_label.setStyleSheet(
            f"color:{accent_pale}; font-family:Consolas; font-size:10px; font-weight:bold;"
        )

        self.preferences_btn.setStyleSheet(
            f"QPushButton {{ background:{accent_deep}; color:{accent_pale}; border:1px solid {accent_value}; padding:7px 10px; }}"
            f"QPushButton:hover {{ background:{accent_dark}; }}"
        )
        self.home_connect_btn.setStyleSheet(
            f"QPushButton {{ background:{accent_dark}; color:#ffffff; border:1px solid {accent_value}; border-radius:4px; padding:7px 14px; font-weight:bold; }}"
            f"QPushButton:hover {{ background:{accent_hover}; }}"
        )
        for nav_button in (self.home_nav_btn,):
            nav_button.setStyleSheet(
                f"QPushButton {{ text-align:left; padding:7px 10px; background:#20282d; border:1px solid #405159; border-radius:4px; }}"
                f"QPushButton:checked {{ background:{accent_deep}; border:2px solid {accent_value}; color:white; font-weight:bold; }}"
                f"QPushButton:hover {{ background:#293a40; border-color:{accent_value}; }}"
            )
        for mode_button in self.mode_buttons:
            mode_button.setStyleSheet(
                f"QPushButton {{ background:#20282d; color:#b5c2c5; border:1px solid #405159;"
                f" border-radius:4px; padding:5px 8px; font-size:10px; font-weight:bold; }}"
                f"QPushButton:checked {{ background:{accent_deep}; color:white; border:2px solid {accent_value}; }}"
                f"QPushButton:hover {{ background:#293a40; border-color:{accent_value}; }}"
            )
        self.home_preview_curve.setPen(pg.mkPen(accent_value, width=1.5))
        for line in self.peak_lines.values():
            line.setPen(pg.mkPen(accent_value, width=1, style=Qt.DashLine))
        for label in self.peak_labels.values():
            label.setColor(accent_value)

    def _theme_color(self, color):
        if self.ui_theme == "Monochrome" and color.upper() in {
            "#4EC9B0", "#00B37E", "#68D6BD", "#D9B36C",
        }:
            return "#ffffff"
        return color

    # ══════════════════════════════════════════════════════════════════════════
    # Toolbar
    # ══════════════════════════════════════════════════════════════════════════
    def init_control_toolbar(self):
        toolbar = QToolBar("Top Control Deck")
        toolbar.setMovable(False)
        toolbar.setFixedHeight(56)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(6)
        self.dashboard_toolbar_widgets = []

        brand = QLabel("◈  EEG STUDIO")
        self.brand_label = brand
        brand.setStyleSheet(
            "color:#4EC9B0; font-size:13px; font-weight:bold; letter-spacing:0.8px; padding-right:12px;"
        )
        layout.addWidget(brand)

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setStyleSheet("color:#3e3e42; background:#3e3e42; max-width:1px;")
        layout.addWidget(divider)

        workspace_label = QLabel("SPECTRUM WORKSPACE")
        self.workspace_label = workspace_label
        workspace_label.setObjectName("WorkspaceBadge")
        workspace_label.setMinimumWidth(156)
        workspace_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(workspace_label)

        mode_group_label = QLabel("MODE")
        mode_group_label.setObjectName("ToolbarGroup")
        mode_group_label.setStyleSheet(
            "color:#b8d8d5; font-size:10px; font-weight:bold; letter-spacing:1px;"
        )
        layout.addWidget(mode_group_label)
        self.mode_buttons = []
        mode_button_group = QButtonGroup(self)
        mode_button_group.setExclusive(True)
        for mode, caption in (
            ("FFT", "FFT"),
            ("EEG Raw", "RAW"),
            ("Sensors", "SENS"),
            ("Oscilloscope", "SCOPE"),
        ):
            button = QPushButton(caption)
            button.setCheckable(True)
            button.setProperty("stream_mode", mode)
            button.setChecked(mode == self.workspace_mode)
            button.setToolTip(f"Switch to {mode} stream")
            button.setFixedHeight(30)
            button.clicked.connect(lambda checked=False, selected=mode: self.select_workspace_mode(selected))
            mode_button_group.addButton(button)
            layout.addWidget(button)
            self.mode_buttons.append(button)

        toolbar_divider = QFrame()
        toolbar_divider.setFrameShape(QFrame.VLine)
        toolbar_divider.setStyleSheet("color:#405159; background:#405159; max-width:1px;")
        layout.addWidget(toolbar_divider)

        def toolbar_group(text):
            label = QLabel(text)
            label.setObjectName("ToolbarGroup")
            self.dashboard_toolbar_widgets.append(label)
            layout.addWidget(label)
            return label

        def toolbar_divider():
            divider = QFrame()
            divider.setFrameShape(QFrame.VLine)
            divider.setStyleSheet("color:#405159; background:#405159; max-width:1px;")
            self.dashboard_toolbar_widgets.append(divider)
            layout.addWidget(divider)

        sensors_group_label = toolbar_group("SENSORS")

        sensors_count_label = QLabel("Active:")
        sensors_count_label.setStyleSheet("color:#858585; font-size:10px; font-weight:bold; padding-left:4px;")
        layout.addWidget(sensors_count_label)
        self.dashboard_toolbar_widgets.append(sensors_count_label)

        self.sensor_count_spin = QSpinBox()
        self.sensor_count_spin.setRange(1, NUM_CHANNELS)
        self.sensor_count_spin.setValue(self.active_sensor_count)
        self.sensor_count_spin.setFixedWidth(56)
        self.sensor_count_spin.setToolTip(
            f"Number of active sensors/channels (1–{NUM_CHANNELS}, limited by the "
            "firmware's fixed frame size)"
        )
        self.sensor_count_spin.valueChanged.connect(self.set_active_sensor_count)
        layout.addWidget(self.sensor_count_spin)
        self.dashboard_toolbar_widgets.append(self.sensor_count_spin)

        # These three widgets are only shown while the Custom Sensors workspace
        # is active; _set_custom_controls_visible() toggles them.
        self.custom_only_widgets = [sensors_group_label, sensors_count_label, self.sensor_count_spin]

        toolbar_divider()
        toolbar_group("LAYOUT")

        self.split_btn = QPushButton("Split View")
        self.split_btn.setFixedHeight(30)
        self.split_btn.clicked.connect(self.toggle_split_view)
        layout.addWidget(self.split_btn)
        self.dashboard_toolbar_widgets.append(self.split_btn)

        self.revert_layout_btn = QPushButton("Reset Order")
        self.revert_layout_btn.clicked.connect(self.revert_channels_to_original)
        self.revert_layout_btn.setStyleSheet(
            "QPushButton { background-color:#202c31; border:1px solid #405159; }"
            "QPushButton:hover { background-color:#293a40; border-color:#46b8a8; }"
        )
        self.revert_layout_btn.setEnabled(False)
        layout.addWidget(self.revert_layout_btn)
        self.dashboard_toolbar_widgets.append(self.revert_layout_btn)

        # Column count for split view
        cols_label = QLabel("Cols:")
        layout.addWidget(cols_label)
        self.dashboard_toolbar_widgets.append(cols_label)
        self.col_spin = QSpinBox()
        self.col_spin.setRange(1, 4)
        self.col_spin.setValue(2)
        self.col_spin.setFixedWidth(48)
        self.col_spin.valueChanged.connect(self._on_col_count_changed)
        layout.addWidget(self.col_spin)
        self.dashboard_toolbar_widgets.append(self.col_spin)

        toolbar_divider()
        toolbar_group("DISPLAY")

        self.freeze_btn = QPushButton("⏸ Freeze")
        self.freeze_btn.setFixedHeight(30)
        self.freeze_btn.setStyleSheet(
            "QPushButton { background-color:#252526; border:1px solid #3e3e42; color:#dcdcaa; }"
            "QPushButton:hover { background-color:#293a40; border-color:#46b8a8; }"
        )
        self.freeze_btn.clicked.connect(self.toggle_freeze)
        layout.addWidget(self.freeze_btn)
        self.dashboard_toolbar_widgets.append(self.freeze_btn)

        self.clear_btn = QPushButton("Clear display")
        self.clear_btn.setFixedHeight(30)
        self.clear_btn.clicked.connect(self.clear_display_data)
        layout.addWidget(self.clear_btn)
        self.dashboard_toolbar_widgets.append(self.clear_btn)

        self.reset_view_btn = QPushButton("Reset view")
        self.reset_view_btn.setFixedHeight(30)
        self.reset_view_btn.clicked.connect(self.reset_dashboard_view)
        layout.addWidget(self.reset_view_btn)
        self.dashboard_toolbar_widgets.append(self.reset_view_btn)

        toolbar_divider()
        toolbar_group("ANALYSIS")

        self.band_btn = QPushButton("⚡ Bands ON")
        self.band_btn.setFixedHeight(30)
        self.band_btn.setStyleSheet(
            "QPushButton { background-color:#164a49; border:1px solid #46b8a8; color:white; }"
            "QPushButton:hover { background-color:#20524e; }"
        )
        self.band_btn.clicked.connect(self.toggle_bands)
        layout.addWidget(self.band_btn)
        self.dashboard_toolbar_widgets.append(self.band_btn)

        self.peak_btn = QPushButton("⌖ Peaks ON")
        self.peak_btn.setFixedHeight(30)
        self.peak_btn.clicked.connect(self.toggle_peak_markers)
        layout.addWidget(self.peak_btn)
        self.dashboard_toolbar_widgets.append(self.peak_btn)

        # These only affect the frequency-domain (FFT) view; hidden in the
        # three raw/oscilloscope-style workspaces via _set_fft_controls_visible().
        self.fft_only_widgets = [self.band_btn, self.peak_btn]

        toolbar_divider()
        toolbar_group("CAPTURE")

        self.record_btn = QPushButton("⏺ Record")
        self.record_btn.setFixedHeight(30)
        self.record_btn.setStyleSheet(
            "QPushButton { background-color:#252526; border:1px solid #3e3e42; color:#cccccc; }"
            "QPushButton:hover { background-color:#293a40; border-color:#46b8a8; }"
        )
        self.record_btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_btn)
        self.dashboard_toolbar_widgets.append(self.record_btn)

        self.export_btn = QPushButton("📷 Snapshot")
        self.export_btn.setFixedHeight(30)
        self.export_btn.clicked.connect(self.export_plot_image)
        layout.addWidget(self.export_btn)
        self.dashboard_toolbar_widgets.append(self.export_btn)

        self.sidebar_btn = QPushButton("◀ Hide Panel")
        self.sidebar_btn.setFixedHeight(30)
        self.sidebar_btn.clicked.connect(self.toggle_sidebar)
        layout.addWidget(self.sidebar_btn)
        self.dashboard_toolbar_widgets.append(self.sidebar_btn)

        layout.addStretch()

        self.runtime_label = QLabel("T+ 00:00.00")
        self.runtime_label.setStyleSheet(
            "color:#00B37E; font-family:'Consolas',monospace; font-size:13px; font-weight:bold; margin-right:10px;"
        )
        layout.addWidget(self.runtime_label)

        self.cursor_readout = QLabel("Hover a trace for details")
        self.cursor_readout.setStyleSheet(
            "color:#858585; font-family:'Consolas',monospace; font-size:11px; padding:0 10px;"
        )
        layout.addWidget(self.cursor_readout)

        self.telemetry_label = QLabel("Waiting for hardware — click Scan then Connect.")
        self.telemetry_label.setObjectName("TelemetryBadge")
        self.telemetry_label.setMinimumWidth(260)
        self.telemetry_label.setAlignment(Qt.AlignCenter)
        self.telemetry_label.setStyleSheet(
            "color:#FFB800; background:#3b351d; border:1px solid #665b2a; border-radius:4px;"
            " padding:4px 8px; font-family:'Consolas',monospace; font-size:10px; font-weight:bold;"
        )
        layout.addWidget(self.telemetry_label)
        self.dashboard_toolbar_widgets.extend([
            self.runtime_label, self.cursor_readout, self.telemetry_label
        ])

        toolbar.addWidget(container)

    # ══════════════════════════════════════════════════════════════════════════
    # Left navigation panel
    # ══════════════════════════════════════════════════════════════════════════
    def init_left_navigation_menu(self, parent_layout):
        self.nav_frame = QFrame()
        self.nav_frame.setFixedWidth(274)
        self.nav_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.nav_frame.setStyleSheet(
            "background-color: #20282d; border-radius: 5px; border: 1px solid #334047;"
        )

        nav_layout = QVBoxLayout(self.nav_frame)
        nav_layout.setContentsMargins(14, 16, 14, 16)
        nav_layout.setSpacing(10)

        app_title = QLabel("EEG STUDIO")
        app_title.setStyleSheet(
            "color:#ffffff; font-size:18px; font-weight:bold; letter-spacing:0.6px; padding:2px 0 0;"
        )
        nav_layout.addWidget(app_title)

        app_subtitle = QLabel("32-CHANNEL SIGNAL ANALYZER")
        app_subtitle.setStyleSheet(
            "color:#858585; font-size:9px; font-weight:bold; letter-spacing:0.8px; padding-bottom:6px;"
        )
        nav_layout.addWidget(app_subtitle)
        nav_layout.addWidget(make_separator())

        self.home_nav_btn = QPushButton("⌂  Home")
        self.home_nav_btn.setCheckable(True)
        self.home_nav_btn.setStyleSheet(
            "QPushButton { text-align:left; padding:7px 10px; background:#20282d; border:1px solid #405159; border-radius:4px; }"
            "QPushButton:checked { background:#164a49; border:2px solid #46b8a8; color:white; font-weight:bold; }"
            "QPushButton:hover { background:#293a40; border-color:#46b8a8; }"
        )
        self.home_nav_btn.setFixedHeight(34)
        self.home_nav_btn.clicked.connect(self.show_home)
        nav_layout.addWidget(self.home_nav_btn)

        nav_layout.addWidget(section_label("00  SIGNAL TYPE"))
        mode_card = QFrame()
        mode_card.setObjectName("CurrentModeCard")
        mode_card.setStyleSheet(
            "QFrame#CurrentModeCard { background:#1b252a; border:1px solid #405159; border-radius:5px; }"
        )
        mode_layout = QVBoxLayout(mode_card)
        mode_layout.setContentsMargins(10, 8, 10, 8)
        mode_layout.setSpacing(3)
        mode_caption = QLabel("CURRENT STREAM")
        mode_caption.setStyleSheet(
            "color:#769096; font-size:9px; font-weight:bold; letter-spacing:0.8px;"
        )
        mode_layout.addWidget(mode_caption)
        self.workspace_mode_label = QLabel(self.workspace_mode.upper())
        self.workspace_mode_label.setStyleSheet(
            "color:#68d6bd; font-size:14px; font-weight:bold;"
        )
        mode_layout.addWidget(self.workspace_mode_label)
        mode_hint = QLabel("Change mode in Preferences")
        mode_hint.setStyleSheet("color:#9aadb1; font-size:10px;")
        mode_layout.addWidget(mode_hint)
        nav_layout.addWidget(mode_card)

        nav_layout.addWidget(make_separator())

        # ── HARDWARE LINK ──────────────────────────────────────────────────
        self.dashboard_nav_start_index = nav_layout.count()
        nav_layout.addWidget(section_label("01  HARDWARE LINK"))

        self.hw_status_label = QLabel("⬤  Not connected")
        self.hw_status_label.setStyleSheet(
            "color:#E25C5C; font-family:Consolas; font-size:12px; font-weight:bold;"
        )
        nav_layout.addWidget(self.hw_status_label)

        self.open_conn_btn = QPushButton("🔌  CONNECT HARDWARE")
        self.open_conn_btn.setStyleSheet(
            "QPushButton { background-color:#238f83; border:1px solid #46b8a8;"
            " color:white; font-weight:bold; padding:8px; }"
            "QPushButton:hover { background-color:#2eaa9b; border-color:#46b8a8; }"
        )
        self.open_conn_btn.clicked.connect(self.open_connection_dialog)
        nav_layout.addWidget(self.open_conn_btn)

        nav_layout.addWidget(make_separator())

        # ── DISPLAY GEOMETRY ───────────────────────────────────────────────
        nav_layout.addWidget(section_label("02  DISPLAY GEOMETRY"))

        cal_row = QHBoxLayout()
        cal_row.addWidget(QLabel("Scale:"))
        self.cal_spin = QDoubleSpinBox()
        self.cal_spin.setRange(0.01, 100.0)
        self.cal_spin.setValue(1.0)
        self.cal_spin.setSingleStep(0.1)
        self.cal_spin.valueChanged.connect(self.update_calibration_scalar)
        cal_row.addWidget(self.cal_spin)
        nav_layout.addLayout(cal_row)
        self.fft_only_widgets.append(self.cal_spin)

        nav_layout.addWidget(QLabel("Plot Height:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(110)
        self.size_slider.setMaximum(500)
        self.size_slider.setValue(self.current_plot_height)
        self.size_slider.valueChanged.connect(self.adjust_plot_sizes)
        nav_layout.addWidget(self.size_slider)

        # X axis max
        x_row = QHBoxLayout()
        self.x_max_label = QLabel("X Max (Hz):")
        x_row.addWidget(self.x_max_label)
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(1, SAMPLE_RATE / 2)
        self.x_max_spin.setDecimals(1)
        self.x_max_spin.setValue(SAMPLE_RATE / 2)
        self.x_max_spin.setSingleStep(1.0)
        self.x_max_spin.valueChanged.connect(self.update_axis_ranges)
        x_row.addWidget(self.x_max_spin)
        nav_layout.addLayout(x_row)

        # Y axis max
        y_row = QHBoxLayout()
        self.y_max_label = QLabel("Y Max (mag):")
        y_row.addWidget(self.y_max_label)
        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(0.01, 100.0)
        self.y_max_spin.setDecimals(2)
        self.y_max_spin.setValue(100.0)
        self.y_max_spin.setSingleStep(0.1)
        self.y_max_spin.valueChanged.connect(self.update_axis_ranges)
        y_row.addWidget(self.y_max_spin)
        nav_layout.addLayout(y_row)

        fit_btn = QPushButton("⊡  Fit to Data")
        fit_btn.setStyleSheet(
            "QPushButton { background-color:#252526; border:1px solid #3e3e42; color:#cccccc; }"
            "QPushButton:hover { background-color:#293a40; border-color:#46b8a8; }"
        )
        fit_btn.clicked.connect(self.fit_axes_to_data)
        nav_layout.addWidget(fit_btn)

        auto_scale = QCheckBox("Auto Y scale")
        auto_scale.setChecked(False)
        auto_scale.toggled.connect(self.set_auto_scale)
        self.auto_scale_check = auto_scale
        nav_layout.addWidget(auto_scale)

        preset_section_label = section_label("03  DISPLAY PRESET")
        nav_layout.addWidget(preset_section_label)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Clinical", "Alpha Focus", "Full Spectrum", "Presentation"])
        self.preset_combo.currentTextChanged.connect(self.apply_display_preset)
        nav_layout.addWidget(self.preset_combo)
        self.fft_only_widgets += [preset_section_label, self.preset_combo]

        nav_layout.addWidget(make_separator())

        # ── ELECTRODE MATRIX ───────────────────────────────────────────────
        nav_layout.addWidget(section_label("05  ELECTRODE MATRIX"))

        macro_row = QHBoxLayout()
        all_on  = QPushButton("All On")
        all_off = QPushButton("All Off")
        all_on.clicked.connect(lambda: self.set_all_checkboxes(True))
        all_off.clicked.connect(lambda: self.set_all_checkboxes(False))
        macro_row.addWidget(all_on)
        macro_row.addWidget(all_off)
        nav_layout.addLayout(macro_row)

        scroll = QScrollArea()
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setContentsMargins(0, 5, 0, 5)
        self.scroll_layout.setSpacing(6)

        self.checkboxes = []
        self.channel_name_labels = []
        for i in range(NUM_CHANNELS):
            row_w = QFrame()
            row_w.setObjectName("ChannelRow")
            row_w.setStyleSheet(
                "QFrame#ChannelRow { background:#1b2227; border:1px solid #2d363c; border-radius:5px; }"
                "QFrame#ChannelRow:hover { border-color:#46b8a8; }"
            )
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(6, 5, 6, 5)
            row_l.setSpacing(5)

            cb = QCheckBox()
            cb.setChecked(i < 4)
            cb.stateChanged.connect(self.update_channel_visibility)
            row_l.addWidget(cb)
            self.checkboxes.append(cb)

            ch_badge = QLabel(f"CH{i+1:02d}")
            ch_badge.setStyleSheet(
                "color:#9aadb1; font-family:Consolas; font-size:10px; font-weight:bold; min-width:34px;"
            )
            row_l.addWidget(ch_badge)

            name_label = QLabel(self._channel_label(i))
            name_label.setStyleSheet("color:#e6e6e6; font-size:11px; font-weight:bold;")
            row_l.addWidget(name_label, stretch=1)
            self.channel_name_labels.append(name_label)

            col_btn = QPushButton()
            col_btn.setFixedSize(20, 20)
            col_btn.setCursor(Qt.PointingHandCursor)
            col_btn.setToolTip("Click to change trace color")
            col_btn.setStyleSheet(
                f"background-color:{self.channel_colors[i].name()}; border:1px solid #444; border-radius:4px;"
            )
            col_btn.clicked.connect(lambda _, idx=i: self.pick_custom_color(idx))
            row_l.addWidget(col_btn)
            self.color_buttons.append(col_btn)

            self.scroll_layout.addWidget(row_w)

        scroll.setWidget(scroll_content)
        scroll.setWidgetResizable(True)
        nav_layout.addWidget(scroll, stretch=1)

        # ── SYSTEM ────────────────────────────────────────────────────────
        self.utility_nav_start_index = nav_layout.count()
        nav_layout.addWidget(make_separator())
        nav_layout.addWidget(section_label("06  SYSTEM"))

        self.preferences_btn = QPushButton("⚙  Preferences")
        self.preferences_btn.clicked.connect(self.open_preferences)
        nav_layout.addWidget(self.preferences_btn)

        parent_layout.insertWidget(0, self.nav_frame)

    # ══════════════════════════════════════════════════════════════════════════
    # Plot init
    # ══════════════════════════════════════════════════════════════════════════
    def _hz_to_bin(self, hz):
        return hz / BIN_TO_HZ

    def _x_max_bin(self):
        if not hasattr(self, "x_max_spin"):
            return DISPLAY_BINS
        if self.display_mode == "Raw Data":
            return self.x_max_spin.value()
        return self._hz_to_bin(self.x_max_spin.value())

    def _display_axis(self):
        return self.raw_x_axis if self.display_mode == "Raw Data" else HZ_AXIS

    def _plot_data_matrix(self):
        if self.display_mode == "Raw Data":
            return self._raw_render_matrix
        return self.fft_data_matrix

    def _refresh_channel_labels(self):
        """Re-label the Electrode Matrix rows and any live legend/title
        text after switching in or out of Custom Sensors mode."""
        if not hasattr(self, "channel_name_labels"):
            return
        for i, name_label in enumerate(self.channel_name_labels):
            name_label.setText(self._channel_label(i))
        if self.is_split_view:
            for idx, wrapper in self.individual_widgets.items():
                wrapper.title_bar.setText(f" ☰  {self._channel_label(idx)}  (CH {idx+1:02d})")
        else:
            for idx, curve in self.curves.items():
                if self.plot_legend is not None:
                    self.plot_legend.removeItem(curve)
                    self.plot_legend.addItem(curve, self._channel_label(idx))

    def _channel_label(self, idx):
        """Display name for a channel: EEG 10-20 electrode name, or a generic
        sensor label while the Custom Sensors workspace is active."""
        if self.custom_sensors_mode:
            return f"Sensor {idx + 1:02d}"
        return EEG_10_20_LABELS[idx]

    def _plot_channel_values(self, channel_idx):
        data = self._plot_data_matrix()[channel_idx]
        if self.display_mode != "Raw Data":
            return data

        if self.is_split_view:
            return data

        visible_channels = list(self.curves)
        try:
            lane = visible_channels.index(channel_idx)
        except ValueError:
            lane = channel_idx
        baseline = max(0, len(visible_channels) - lane - 1) * 2 * self._raw_plot_amplitude()
        return baseline + data

    def _raw_plot_amplitude(self):
        return 100.0

    def _y_max(self):
        if self.display_mode == "Raw Data":
            return self._raw_plot_amplitude() if self.is_split_view else max(
                self._raw_plot_amplitude(),
                2 * self._raw_plot_amplitude() * float(len(self.curves))
            )
        return self.y_max_spin.value() if hasattr(self, "y_max_spin") else 80

    def init_plot_display(self):
        self.plot_widget = pg.PlotWidget()
        self._style_plot_widget(self.plot_widget, step_hz=8)

        self.band_region_items = add_band_lines(self.plot_widget, show=self.show_bands)
        self.plot_legend = self.plot_widget.addLegend(offset=(10, 10))
        self.plot_legend.setBrush(pg.mkBrush('#252526E6'))
        self.plot_legend.setPen(pg.mkPen('#3e3e42'))
        self._connect_plot_hover(self.plot_widget)

        self.plot_container_layout.addWidget(self._create_graph_header())
        self.plot_container_layout.addWidget(self.plot_widget)

    def _create_graph_header(self):
        header = QFrame()
        header.setMinimumHeight(52)
        header.setStyleSheet(
            "QFrame { background:#252526; border-bottom:1px solid #3e3e42; }"
            "QLabel#GraphTitle { color:#ffffff; font-size:16px; font-weight:bold; letter-spacing:0.4px; }"
            "QLabel#GraphDetail { color:#9aadb1; font-size:11px; }"
            "QLabel#GraphStatus { color:#68d6bd; font-family:Consolas; font-size:10px; font-weight:bold; }"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_layout.setSpacing(8)

        title = QLabel("LIVE SPECTRUM")
        title.setObjectName("GraphTitle")
        self.graph_title_label = title
        header_layout.addWidget(title)
        detail = QLabel("Magnitude by frequency  |  0–128 Hz")
        detail.setObjectName("GraphDetail")
        self.graph_detail_label = detail
        header_layout.addWidget(detail)
        header_layout.addStretch()
        self.graph_status_label = QLabel("READY")
        self.graph_status_label.setObjectName("GraphStatus")
        header_layout.addWidget(self.graph_status_label)
        return header

    def _connect_plot_hover(self, plot_widget, channel_idx=None):
        proxy = pg.SignalProxy(
            plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=lambda event: self._on_plot_mouse_moved(event, plot_widget, channel_idx),
        )
        self.plot_signal_proxies.append(proxy)

    def _on_plot_mouse_moved(self, event, plot_widget, channel_idx=None):
        if self.cursor_readout is None:
            return
        position = event[0] if isinstance(event, tuple) else event
        view_box = plot_widget.getPlotItem().vb
        if not view_box.sceneBoundingRect().contains(position):
            self.cursor_readout.setText("Hover a trace for details")
            return

        point = view_box.mapSceneToView(position)
        data_matrix = self._plot_data_matrix()
        max_index = data_matrix.shape[1] - 1
        if self.display_mode == "Raw Data":
            # The x-axis is time in seconds; convert back to a sample index.
            bin_index = int(np.clip(round(point.x() * SAMPLE_RATE), 0, max_index))
            horizontal_text = f"{point.x():.3f} s"
        else:
            bin_index = int(np.clip(round(point.x()), 0, max_index))
            horizontal_value = HZ_AXIS[bin_index]
            horizontal_text = f"{horizontal_value:.1f} Hz"
        if channel_idx is None:
            active = list(self.curves.keys())
            if not active:
                self.cursor_readout.setText("No active traces")
                return
            if self.display_mode == "Raw Data":
                channel_idx = max(
                    active,
                    key=lambda idx: abs(data_matrix[idx, bin_index] - 32768.0),
                )
            else:
                channel_idx = max(active, key=lambda idx: data_matrix[idx, bin_index])
        magnitude = data_matrix[channel_idx, bin_index]
        label = self._channel_label(channel_idx)
        if self.display_mode == "Raw Data":
            value_name = ADS1299_RAW_UNIT
            self.cursor_readout.setText(
                f"{label}  |  {horizontal_text}  |  {magnitude:.3f} {value_name}"
            )
        else:
            value_name = "mag"
            self.cursor_readout.setText(
                f"{label}  |  {horizontal_text}  |  {magnitude:.2f} {value_name}"
            )

    def _ensure_peak_marker(self, channel_idx, plot_widget):
        key = (id(plot_widget), channel_idx)
        if key in self.peak_lines:
            return
        line = InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen('#00B37E', width=1, style=Qt.DashLine),
        )
        line.setVisible(self.show_peak_markers)
        plot_widget.addItem(line)
        label = pg.TextItem(color='#00B37E', anchor=(0.5, 1.0))
        label.setVisible(self.show_peak_markers)
        plot_widget.addItem(label)
        self.peak_lines[key] = line
        self.peak_labels[key] = label

    def _update_peak_marker(self, channel_idx, data_row, plot_widget):
        if self.display_mode == "Raw Data":
            return
        key = (id(plot_widget), channel_idx)
        self._ensure_peak_marker(channel_idx, plot_widget)
        peak_bin = int(np.argmax(data_row))
        peak_value = float(data_row[peak_bin])
        self.peak_lines[key].setPos(peak_bin)
        if self.display_mode == "Raw Data":
            self.peak_labels[key].setText(f"bin {peak_bin:03d}")
        else:
            self.peak_labels[key].setText(f"{peak_bin * BIN_TO_HZ:.1f} Hz")
        self.peak_labels[key].setPos(peak_bin, max(peak_value, self._y_max() * 0.92))

    def toggle_peak_markers(self):
        self.show_peak_markers = not self.show_peak_markers
        self.peak_btn.setText("⌖ Peaks ON" if self.show_peak_markers else "⌖ Peaks OFF")
        for item in list(self.peak_lines.values()) + list(self.peak_labels.values()):
            item.setVisible(self.show_peak_markers and self.display_mode != "Raw Data")

    def _mode_geometry_config(self):
        if self.display_mode == "Raw Data":
            full_window_s = round(RAW_WINDOW_BINS / SAMPLE_RATE, 2)
            return {
                "x_label": "Time span (s):",
                "x_min": round(1.0 / SAMPLE_RATE, 3),
                "x_max": full_window_s,
                "x_decimals": 2,
                "x_step": 0.05,
                "x_value": full_window_s,
                "y_label": f"Y Max ({ADS1299_RAW_UNIT}):",
                "y_min": 10.0,
                "y_max": 10000.0,
                "y_decimals": 0,
                "y_step": 10.0,
                "y_value": 100.0,
            }
        return {
            "x_label": "X Max (Hz):",
            "x_min": 1.0,
            "x_max": float(SAMPLE_RATE / 2),
            "x_decimals": 1,
            "x_step": 1.0,
            "x_value": float(SAMPLE_RATE / 2),
            "y_label": f"Y Max ({ADS1299_RAW_UNIT}):",
            "y_min": 1.0,
            "y_max": 10000.0,
            "y_decimals": 0,
            "y_step": 10.0,
            "y_value": 100.0,
        }

    def _style_plot_widget(self, plot_widget, step_hz):
        plot_widget.setBackground('#202024')
        left_label = ADS1299_RAW_UNIT if self.display_mode == "Raw Data" else f"Magnitude ({ADS1299_RAW_UNIT})"
        plot_widget.setLabel('left', left_label, **{'color': '#858585', 'font-size': '11px'})
        bottom_label = 'Time (s)' if self.display_mode == "Raw Data" else 'Frequency (Hz)'
        plot_widget.setLabel('bottom', bottom_label, **{'color': '#858585', 'font-size': '11px'})
        plot_item = plot_widget.getPlotItem()
        pg.setConfigOptions(antialias=False)
        plot_item.setDownsampling(auto=False)
        plot_item.setClipToView(True)
        plot_widget.showGrid(x=True, y=True, alpha=0.10)
        plot_widget.setXRange(0, self._x_max_bin(), padding=0)
        plot_widget.setYRange(0, self._y_max(), padding=0)
        plot_widget.setMouseEnabled(x=False, y=True)
        plot_widget.setMenuEnabled(False)

        for axis_name in ('left', 'bottom'):
            axis = plot_widget.getAxis(axis_name)
            axis.setPen(pg.mkPen('#3e3e42'))
            axis.setTextPen(pg.mkPen('#858585'))
            axis.setStyle(tickTextOffset=6)

        plot_widget.getAxis('bottom').setTicks([self._build_hz_ticks(step_hz=step_hz)])

    def _build_hz_ticks(self, step_hz=8):
        """Return major tick list [(position, label), ...].

        In Raw Data (oscilloscope) mode, positions/labels are in seconds.
        In FFT mode, positions are bin indices labeled with their Hz value.
        """
        ticks = []
        if self.display_mode == "Raw Data":
            total_seconds = RAW_WINDOW_BINS / SAMPLE_RATE
            step_s = max(0.01, round(total_seconds / 8, 2))
            t = 0.0
            while t <= total_seconds + 1e-9:
                ticks.append((t, f"{t:.2f}s"))
                t += step_s
            return ticks
        hz = 0
        while hz <= SAMPLE_RATE / 2:
            ticks.append((self._hz_to_bin(hz), f"{hz}"))
            hz += step_hz
        return ticks

    def set_display_mode(self, mode):
        self.display_mode = mode
        # Renderer selection is separate from the STM32 workspace command.
        # select_workspace_mode() owns the protocol mode byte.
        raw_mode = mode == "Raw Data"
        if raw_mode:
            self._raw_ring_write = 0
            self._raw_ring_count = 0
            self._raw_new_samples = 0
            self.raw_data_matrix.fill(0.0)
            self._raw_render_matrix.fill(0.0)
        cfg = self._mode_geometry_config()

        self.x_max_label.setText(cfg["x_label"])
        self.y_max_label.setText(cfg["y_label"])

        self.x_max_spin.blockSignals(True)
        self.x_max_spin.setRange(cfg["x_min"], cfg["x_max"])
        self.x_max_spin.setDecimals(cfg["x_decimals"])
        self.x_max_spin.setSingleStep(cfg["x_step"])
        self.x_max_spin.setValue(cfg["x_value"])
        self.x_max_spin.blockSignals(False)

        self.y_max_spin.blockSignals(True)
        self.y_max_spin.setRange(cfg["y_min"], cfg["y_max"])
        self.y_max_spin.setDecimals(cfg["y_decimals"])
        self.y_max_spin.setSingleStep(cfg["y_step"])
        self.y_max_spin.setValue(cfg["y_value"])
        self.y_max_spin.blockSignals(False)

        plots = [self.plot_widget] if not self.is_split_view else [
            wrapper.plot_widget for wrapper in self.individual_widgets.values()
        ]
        for plot_widget in plots:
            plot_widget.setLabel(
                'left',
                ADS1299_RAW_UNIT if raw_mode else f'Magnitude ({ADS1299_RAW_UNIT})',
                **{'color': '#858585', 'font-size': '11px'},
            )
            plot_widget.setLabel(
                'bottom',
                'Time (s)' if raw_mode else 'Frequency (Hz)',
                **{'color': '#858585', 'font-size': '11px'},
            )
            plot_widget.setXRange(0, self._x_max_bin(), padding=0)
            plot_widget.getAxis('bottom').setTicks([self._build_hz_ticks()])

        self.graph_title_label.setText("RAW OSCILLOSCOPE" if raw_mode else "LIVE SPECTRUM")
        self.graph_detail_label.setText(
            f"Analog waveform  |  {ADS1299_RAW_UNIT} vs Time (s)"
            if raw_mode else "Magnitude by frequency  |  0–128 Hz"
        )
        self.plot_legend.setVisible(not raw_mode)
        for item in list(self.peak_lines.values()) + list(self.peak_labels.values()):
            item.setVisible(self.show_peak_markers and not raw_mode)
        self._apply_band_visibility()
        display_axis = self._display_axis()
        for idx, curve in self.curves.items():
            curve.setPen(pg.mkPen(color=self.channel_colors[idx], width=1 if raw_mode else 2))
            # FFT curves have 128 x-values; raw/oscilloscope views render the
            # 512-point time window and must replace the x-axis on mode switch.
            curve.setData(
                x=display_axis,
                y=self._plot_channel_values(idx),
                skipFiniteCheck=True,
            )
        self.update_axis_ranges()

    def _send_stream_command(self, command):
        if not self.ser or not self.ser.is_open:
            return False
        try:
            payload = command.encode('utf-8') if isinstance(command, str) else command
            self.ser.write(payload)
            self.ser.flush()
        except Exception as error:
            print(f"ACTUAL ERROR: {error}") # This will print the real reason to your terminal
            self._set_status("Stream command failed", "#E25C5C")
            return False
        return True

    def _configure_split_plot(self, pw):
        self._style_plot_widget(pw, step_hz=16)
        add_band_lines(pw, show=self.show_bands)

    # ══════════════════════════════════════════════════════════════════════════
    # Toolbar actions
    # ══════════════════════════════════════════════════════════════════════════
    def toggle_freeze(self):
        self.is_frozen = not self.is_frozen
        if self.is_frozen:
            self.freeze_btn.setText("▶ Resume")
            self.graph_status_label.setText("FROZEN")
            self.freeze_btn.setStyleSheet(
                "QPushButton { background-color:#164a49; border:1px solid #46b8a8; color:white; }"
                "QPushButton:hover { background-color:#20524e; }"
            )
        else:
            self.freeze_btn.setText("⏸ Freeze")
            self.graph_status_label.setText("LIVE")
            self.freeze_btn.setStyleSheet(
                "QPushButton { background-color:#202c31; border:1px solid #405159; color:#d9b36c; }"
                "QPushButton:hover { background-color:#293a40; border-color:#46b8a8; }"
            )

    def clear_display_data(self):
        self.fft_data_matrix.fill(0.0)
        self.raw_data_matrix.fill(0.0)
        self._raw_render_matrix.fill(0.0)
        self._raw_ring_write = 0
        self._raw_ring_count = 0
        self._raw_new_samples = 0
        self._frame_ready = False
        if hasattr(self, "home_preview_curve"):
            self.home_preview_curve.setData(
                x=self._display_axis(),
                y=np.zeros(len(self._display_axis()), dtype=np.float32),
            )
        self._set_status("Display cleared.", "#d9b36c")

    def reset_dashboard_view(self):
        self.auto_scale_y = False
        self.auto_scale_check.setChecked(False)
        self.apply_display_preset("Clinical")
        if not self.show_bands:
            self.toggle_bands()
        if not self.show_peak_markers:
            self.toggle_peak_markers()
        if self.is_split_view:
            self.revert_channels_to_original()
        self.update_axis_ranges()
        self._set_status("Dashboard view reset.", "#68d6bd")

    def toggle_bands(self):
        self.show_bands = not self.show_bands
        self.band_btn.setText("⚡ Bands ON" if self.show_bands else "⚡ Bands OFF")
        self._apply_band_visibility()

    def _apply_band_visibility(self):
        plots = [self.plot_widget] if not self.is_split_view else [
            wrapper.plot_widget for wrapper in self.individual_widgets.values()
        ]
        for plot_widget in plots:
            for item in plot_widget.items():
                band_index = getattr(item, '_eeg_band_index', None)
                if band_index is not None:
                    item.setVisible(self.display_mode != "Raw Data" and self.show_bands)

    def toggle_sidebar(self):
        self.nav_visible = not self.nav_visible
        self.nav_frame.setVisible(self.nav_visible)
        self.sidebar_btn.setText("◀ Hide Panel" if self.nav_visible else "▶ Show Panel")

    def toggle_recording(self):
        if not self._recording:
            dlg = RecordingDialog(
                [self._channel_label(i) for i in range(NUM_CHANNELS)], parent=self
            )
            dlg.set_active_channels(list(self.curves.keys()))
            if dlg.exec_() != QDialog.Accepted:
                return
            path = dlg.output_path
            if not path:
                return

            self._record_channels = dlg.selected_channels
            self._record_duration = dlg.duration_seconds
            self._record_elapsed  = 0.0

            try:
                self._csv_file = open(path, 'w', newline='')
            except OSError as error:
                self._set_status(f"Recording failed: {error}", "#E25C5C")
                return
            self._csv_writer = csv.writer(self._csv_file)
            value_prefix = "raw" if self.display_mode == "Raw Data" else "bin"
            sample_count = RAW_WINDOW_BINS if self.display_mode == "Raw Data" else DISPLAY_BINS
            header = ["timestamp"] + [
                f"{self._channel_label(ch)}_{value_prefix}{b}"
                for ch in self._record_channels
                for b in range(sample_count)
            ]
            self._csv_writer.writerow(header)
            self._recording = True

            # Countdown label suffix
            if self._record_duration:
                self.record_btn.setText(f"⏹ {self._record_duration}s left")
            else:
                self.record_btn.setText("⏹ Stop Rec")
            self.record_btn.setStyleSheet(
                "QPushButton { background-color:#164a49; border:1px solid #46b8a8; color:white; }"
                "QPushButton:hover { background-color:#20524e; }"
            )
        else:
            self._stop_recording()

    def _stop_recording(self):
        self._recording = False
        if self._csv_file:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None
        self.record_btn.setText("⏺ Record")
        self.record_btn.setStyleSheet(
            "QPushButton { background-color:#252526; border:1px solid #3e3e42; color:#cccccc; }"
            "QPushButton:hover { background-color:#293a40; border-color:#46b8a8; }"
        )

    def export_plot_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", "eeg_snapshot.png", "PNG Image (*.png)"
        )
        if path:
            target = self.split_scroll.viewport() if self.is_split_view else self.plot_widget
            if not target.grab().save(path, "PNG"):
                self.telemetry_label.setText("Export Failed")

    def _on_col_count_changed(self, val):
        self.split_columns = val
        if self.is_split_view:
            self._rebuild_grid_positions()

    def _rebuild_grid_positions(self):
        active_count = 0
        for idx in self.channel_render_order:
            if self.checkboxes[idx].isChecked() and idx in self.individual_widgets:
                w   = self.individual_widgets[idx]
                row = active_count // self.split_columns
                col = active_count %  self.split_columns
                self.grid_layout.removeWidget(w)
                self.grid_layout.addWidget(w, row, col)
                active_count += 1

    # ══════════════════════════════════════════════════════════════════════════
    # Axis / display controls
    # ══════════════════════════════════════════════════════════════════════════
    def adjust_plot_sizes(self, val):
        self.current_plot_height = val
        if self.is_split_view:
            for w in self.individual_widgets.values():
                w.setFixedHeight(val)

    def update_axis_ranges(self):
        x_bin = self._x_max_bin()
        y_max = self._y_max()
        y_min = -y_max if self.display_mode == "Raw Data" else 0
        if self.is_split_view:
            for w in self.individual_widgets.values():
                w.plot_widget.setXRange(0, x_bin, padding=0)
                w.plot_widget.setYRange(y_min, y_max, padding=0)
        else:
            self.plot_widget.setXRange(0, x_bin, padding=0)
            self.plot_widget.setYRange(y_min, y_max, padding=0)

    def set_auto_scale(self, enabled):
        self.auto_scale_y = enabled
        if enabled:
            self._apply_auto_y_range()

    def _apply_auto_y_range(self):
        active = list(self.curves.keys())
        if not active:
            return
        if self.display_mode == "Raw Data":
            self.update_axis_ranges()
            return
        maximum = float(np.max(self.fft_data_matrix[active]))
        y_max = max(1.0, maximum * 1.12)
        plots = [self.plot_widget] if not self.is_split_view else [
            wrapper.plot_widget for wrapper in self.individual_widgets.values()
        ]
        for plot_widget in plots:
            plot_widget.setYRange(0, y_max, padding=0)

    def apply_display_preset(self, preset_name):
        if self.display_mode == "Raw Data":
            full_window_s = round(RAW_WINDOW_BINS / SAMPLE_RATE, 2)
            quarter_window_s = round(full_window_s / 4, 2)
            presets = {
                "Clinical": (quarter_window_s, 100.0, 180),
                "Alpha Focus": (quarter_window_s, 100.0, 210),
                "Full Spectrum": (full_window_s, 100.0, 180),
                "Presentation": (quarter_window_s, 100.0, 260),
            }
        else:
            presets = {
                "Clinical": (45.0, 1.0, 180),
                "Alpha Focus": (20.0, 1.0, 210),
                "Full Spectrum": (SAMPLE_RATE / 2, 1.0, 180),
                "Presentation": (45.0, 1.0, 260),
            }
        x_max, y_max, plot_height = presets.get(preset_name, presets["Clinical"])
        self.x_max_spin.blockSignals(True)
        self.y_max_spin.blockSignals(True)
        self.x_max_spin.setValue(x_max)
        self.y_max_spin.setValue(y_max)
        self.x_max_spin.blockSignals(False)
        self.y_max_spin.blockSignals(False)
        self.size_slider.setValue(plot_height)
        if not self.show_bands:
            self.toggle_bands()

    def fit_axes_to_data(self):
        active = list(self.curves.keys())
        if not active:
            return
        if self.display_mode == "Raw Data":
            plotted = np.vstack([
                self._plot_channel_values(idx) for idx in active
            ])
            data_min = float(np.min(plotted))
            data_max = float(np.max(plotted))
            span = max(data_max - data_min, 1.0)
            padding = span * 0.06
            y_min = data_min - padding
            y_max = data_max + padding
            self.x_max_spin.blockSignals(True)
            self.y_max_spin.blockSignals(True)
            self.x_max_spin.setValue(round(RAW_WINDOW_BINS / SAMPLE_RATE, 2))
            self.y_max_spin.setValue(round(max(abs(y_min), abs(y_max)), 2))
            self.x_max_spin.blockSignals(False)
            self.y_max_spin.blockSignals(False)
            if self.is_split_view:
                for idx, wrapper in self.individual_widgets.items():
                    if idx not in active:
                        continue
                    values = self._plot_channel_values(idx)
                    local_min = float(np.min(values))
                    local_max = float(np.max(values))
                    local_padding = max((local_max - local_min) * 0.06, 1.0)
                    wrapper.plot_widget.setXRange(0, self._x_max_bin(), padding=0)
                    wrapper.plot_widget.setYRange(
                        local_min - local_padding,
                        local_max + local_padding,
                        padding=0,
                    )
            else:
                self.plot_widget.setXRange(0, self._x_max_bin(), padding=0)
                self.plot_widget.setYRange(y_min, y_max, padding=0)
            return
        visible = self.fft_data_matrix[active]
        y_min   = float(visible.min())
        y_max   = float(visible.max()) + max((float(visible.max()) - float(visible.min())) * 0.05, 1.0)

        self.x_max_spin.blockSignals(True)
        self.y_max_spin.blockSignals(True)
        self.x_max_spin.setValue(RAW_WINDOW_BINS if self.display_mode == "Raw Data" else SAMPLE_RATE / 2)
        if self.display_mode == "Raw Data":
            self.y_max_spin.setValue(round(max(y_max, 1.0), 2))
        else:
            self.y_max_spin.setValue(round(y_max))
        self.x_max_spin.blockSignals(False)
        self.y_max_spin.blockSignals(False)

        x_bin = self._x_max_bin()
        if self.is_split_view:
            for w in self.individual_widgets.values():
                w.plot_widget.setXRange(0, x_bin, padding=0)
                w.plot_widget.setYRange(y_min, y_max, padding=0)
        else:
            self.plot_widget.setXRange(0, x_bin, padding=0)
            self.plot_widget.setYRange(y_min, y_max, padding=0)

    def update_calibration_scalar(self, val):
        self.calibration_scalar = val

    # ══════════════════════════════════════════════════════════════════════════
    # Channel visibility / drag-drop
    # ══════════════════════════════════════════════════════════════════════════
    def set_active_drag_origin(self, ch_idx):
        self.active_drag_origin_idx = ch_idx

    def set_all_checkboxes(self, state):
        for cb in self.checkboxes:
            cb.blockSignals(True)
            cb.setChecked(state)
            cb.blockSignals(False)
        self.update_channel_visibility()

    def pick_custom_color(self, ch_idx):
        color = QColorDialog.getColor(self.channel_colors[ch_idx], self, "Choose Trace Color")
        if color.isValid():
            self.channel_colors[ch_idx] = color
            self.color_buttons[ch_idx].setStyleSheet(
                f"background-color:{color.name()}; border:1px solid #fff; border-radius:2px;"
            )
            if ch_idx in self.curves:
                self.curves[ch_idx].setPen(pg.mkPen(color=color, width=2))

    def set_active_sensor_count(self, n):
        """Custom Sensors mode: activate the first n channels (1..NUM_CHANNELS)
        and deactivate the rest. Channel count itself is capped at NUM_CHANNELS
        because that is what the firmware's fixed-size frame protocol supports;
        this control lets you dial how many of those inputs are treated as
        live sensors, from a single channel up to the full set."""
        if not hasattr(self, "checkboxes"):
            return
        n = int(np.clip(n, 1, NUM_CHANNELS))
        self.active_sensor_count = n
        for i, cb in enumerate(self.checkboxes):
            cb.blockSignals(True)
            cb.setChecked(i < n)
            cb.blockSignals(False)
        self.update_channel_visibility()
        if getattr(self, "workspace_mode", "") == "Sensors":
            self._refresh_workspace_ui()
        self._set_status(f"Sensors: {n} of {NUM_CHANNELS} channel(s) active.", self.ui_accent)

    def update_channel_visibility(self):
        active_count = 0
        for idx in self.channel_render_order:
            cb = self.checkboxes[idx]
            if cb.isChecked():
                if self.is_split_view:
                    if self.grid_layout is None:
                        continue
                    if idx not in self.individual_widgets:
                        pw = pg.PlotWidget()
                        self._configure_split_plot(pw)
                        self._connect_plot_hover(pw, idx)
                        wrapper = DraggablePlotWrapper(
                            idx, pw, initial_height=self.current_plot_height,
                            channel_label=self._channel_label(idx),
                        )
                        row = active_count // self.split_columns
                        col = active_count %  self.split_columns
                        self.grid_layout.addWidget(wrapper, row, col)
                        self.individual_widgets[idx] = wrapper
                        self.curves[idx] = pw.plot(
                            x=self._display_axis(),
                            y=self._plot_channel_values(idx),
                            pen=pg.mkPen(color=self.channel_colors[idx], width=2),
                        )
                        self._ensure_peak_marker(idx, pw)
                    else:
                        if not self.is_animating_layout:
                            w   = self.individual_widgets[idx]
                            row = active_count // self.split_columns
                            col = active_count %  self.split_columns
                            self.grid_layout.addWidget(w, row, col)
                    active_count += 1
                else:
                    if idx not in self.curves:
                        self.curves[idx] = self.plot_widget.plot(
                            x=self._display_axis(),
                            y=self._plot_channel_values(idx),
                            pen=pg.mkPen(color=self.channel_colors[idx], width=2),
                        )
                        self.plot_legend.addItem(
                            self.curves[idx], self._channel_label(idx)
                        )
                        self._ensure_peak_marker(idx, self.plot_widget)
            else:
                if self.is_split_view and idx in self.individual_widgets:
                    if self.grid_layout is not None:
                        self.grid_layout.removeWidget(self.individual_widgets[idx])
                    self.individual_widgets[idx].setParent(None)
                    del self.individual_widgets[idx]
                    if idx in self.curves:
                        del self.curves[idx]
                elif not self.is_split_view and idx in self.curves:
                    self.plot_widget.removeItem(self.curves[idx])
                    self.plot_legend.removeItem(self.curves[idx])
                    del self.curves[idx]

    # ══════════════════════════════════════════════════════════════════════════
    # Hardware
    # ══════════════════════════════════════════════════════════════════════════
    def open_connection_dialog(self):
        dlg = ConnectionDialog(self.ser, parent=self)
        dlg.exec_()

        self.ser = dlg.serial_port

        if self.ser and self.ser.is_open:
            # Every connection change starts a clean protocol session.
            self.ser.reset_input_buffer()
            self._reset_acquisition_session()
            self.session_runtime = 0.0
            self.last_update_time = time.time()
            self._last_sent_stream_command = None   # force resend on this fresh link
            self._send_workspace_command()
            port = dlg.selected_port_name
            baud = dlg.selected_baud
            self.hw_status_label.setText(f"⬤  {port} @ {baud}")
            self.hw_status_label.setStyleSheet(
                "color:#00B37E; font-family:Consolas; font-size:12px; font-weight:bold;"
            )
            self.open_conn_btn.setText("🔌  Manage Connection")
            self._set_status(f"Connected: {port} @ {baud}", "#00B37E")
        else:
            self._serial_buf.clear()
            self._last_sent_stream_command = None
            self.hw_status_label.setText("⬤  Not connected")
            self.hw_status_label.setStyleSheet(
                "color:#E25C5C; font-family:Consolas; font-size:12px; font-weight:bold;"
            )
            self.open_conn_btn.setText("🔌  Open Connection Manager")
            self._set_status("Disconnected — waiting for hardware.", "#FFB800")

    def _set_status(self, text, color):
        self.telemetry_label.setText(text)
        self.telemetry_label.setStyleSheet(
            f"color:{color}; font-family:'Consolas'; font-size:12px; font-weight:bold;"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Split-view / drag-drop machinery
    # ══════════════════════════════════════════════════════════════════════════
    def toggle_split_view(self):
        self.is_split_view = not self.is_split_view

        for i in reversed(range(self.plot_container_layout.count())):
            w = self.plot_container_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        self.curves.clear()
        self.individual_widgets.clear()
        self.band_region_items.clear()
        self.plot_signal_proxies.clear()
        self.peak_lines.clear()
        self.peak_labels.clear()

        if self.is_split_view:
            self.split_btn.setText("Merge View")
            self.revert_layout_btn.setEnabled(True)
            self.split_scroll = QScrollArea()
            self.split_scroll.setWidgetResizable(True)
            scroll_content = QWidget()
            scroll_content.setStyleSheet("background-color:#121214;")
            scroll_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.grid_layout = RearrangeableGridLayout(scroll_content)
            self.grid_layout.setSpacing(10)
            self.grid_layout.setSizeConstraint(QGridLayout.SetMinAndMaxSize)
            scroll_content.setAcceptDrops(True)
            scroll_content.dragEnterEvent = self.grid_layout.handle_drag_enter
            scroll_content.dragMoveEvent  = self.grid_layout.handle_drag_move
            scroll_content.dropEvent      = self.grid_layout.handle_drop
            self.split_scroll.setWidget(scroll_content)
            self.plot_container_layout.addWidget(self._create_graph_header())
            self.plot_container_layout.addWidget(self.split_scroll)
        else:
            self.split_btn.setText("Split View")
            self.revert_layout_btn.setEnabled(False)
            self.grid_layout = None
            self.init_plot_display()

        self.update_channel_visibility()

    def live_swap_channels(self, source, target):
        if self.is_animating_layout or self.grid_layout is None:
            return
        si = self.channel_render_order.index(source)
        ti = self.channel_render_order.index(target)
        tw = self.individual_widgets.get(target)
        sw = self.individual_widgets.get(source)
        if not tw or not sw:
            return
        t_orig = tw.pos()
        self.channel_render_order[si], self.channel_render_order[ti] = \
            self.channel_render_order[ti], self.channel_render_order[si]
        self.is_animating_layout = True
        self.update_grid_positions_with_animation(source, target, t_orig)

    def revert_channels_to_original(self):
        if self.grid_layout is None:
            return
        self.channel_render_order = list(range(NUM_CHANNELS))
        self.is_animating_layout = False
        self._rebuild_grid_positions()
        self._set_status("Channel order restored.", self.ui_accent)

    def update_grid_positions_with_animation(self, dragging_id, bumped_id, target_pre_pos):
        if self.grid_layout is None:
            return
        active_count = 0
        for idx in self.channel_render_order:
            if self.checkboxes[idx].isChecked() and idx in self.individual_widgets:
                w    = self.individual_widgets[idx]
                row  = active_count // self.split_columns
                col  = active_count %  self.split_columns
                dest = self.grid_layout.cellRect(row, col).topLeft()
                if idx == bumped_id:
                    w.animate_to_pos(dest)
                    w.trigger_drop_flash()
                elif idx == dragging_id:
                    w.move(dest)
                else:
                    if w.pos() != dest:
                        w.move(dest)
                active_count += 1
        QTimer.singleShot(260, self.re_engage_grid_layout)

    def re_engage_grid_layout(self):
        if self.grid_layout is None:
            return
        active_count = 0
        for idx in self.channel_render_order:
            if self.checkboxes[idx].isChecked() and idx in self.individual_widgets:
                w   = self.individual_widgets[idx]
                row = active_count // self.split_columns
                col = active_count %  self.split_columns
                self.grid_layout.removeWidget(w)
                self.grid_layout.addWidget(w, row, col)
                active_count += 1
        self.is_animating_layout = False

    def finalize_drop_execution(self, source_idx):
        if source_idx in self.individual_widgets:
            self.individual_widgets[source_idx].trigger_drop_flash()
        self.active_drag_origin_idx = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    # ══════════════════════════════════════════════════════════════════════════
    # Main update loop
    # ══════════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════════
    # Binary frame parser
    # ══════════════════════════════════════════════════════════════════════════
    def _parse_serial_frames(self):
        """
        Consume as many complete frames as possible from _serial_buf.

        Frame layout (matches STM32 firmware):
          [0xAA][0x55]  sync
          [seq lo][seq hi]  uint16 LE
          [nch]  uint8
          [nbins lo][nbins hi]  uint16 LE
          [data]  nch * nbins * uint16 LE  (magnitudes 0-65535)
          [crc lo][crc hi]  uint16 LE  CRC-16/CCITT over data only

        Sync-hunt: scan forward byte-by-byte until 0xAA 0x55 found.
        """
        buf = self._serial_buf

        while True:
            # Hunt for sync
            if len(buf) < 2:
                break
            sync_pos = -1
            for i in range(len(buf) - 1):
                if buf[i] == PROTO_SYNC0 and buf[i+1] == PROTO_SYNC1:
                    sync_pos = i
                    break
            if sync_pos == -1:
                # No sync found — keep last byte in case it's the first sync byte
                self._serial_buf = bytearray(buf[-1:])
                return
            if sync_pos > 0:
                # Discard garbage before sync
                buf = buf[sync_pos:]

            if len(buf) < PROTO_FRAME_SIZE:
                break   # Wait for more bytes

            # Parse header
            seq    = struct.unpack_from('<H', buf, 2)[0]
            nch    = buf[4]
            nbins  = struct.unpack_from('<H', buf, 5)[0]

            if nch != NUM_CHANNELS or nbins != DISPLAY_BINS:
                # Header mismatch — skip this sync and keep hunting
                buf = buf[2:]
                continue

            # Extract data region and CRC
            data_start = PROTO_HDR_SIZE
            data_end   = data_start + PROTO_DATA_BYTES
            data_bytes = buf[data_start:data_end]
            rx_crc     = struct.unpack_from('<H', buf, data_end)[0]
            calc_crc   = self._crc16(data_bytes)

            if rx_crc != calc_crc:
                self._crc_errors += 1
                self.dropped_packets += 1
                buf = buf[2:]   # skip sync, try next
                continue

            # Reject duplicates and old frames before updating stream state.
            if self._last_seq >= 0:
                sequence_delta = (seq - self._last_seq) & 0xFFFF
                if sequence_delta == 0 or sequence_delta > 0x8000:
                    buf = buf[PROTO_FRAME_SIZE:]
                    continue
                if sequence_delta > 1:
                    self.dropped_packets += sequence_delta - 1

            # Good frame
            self.packet_count += 1
            self._last_seq = seq

            if self.display_mode == "Raw Data":
                # The placeholder firmware uses signed int16 microvolts for its
                # ADS1299-style time-domain stream.
                raw_frame = np.frombuffer(data_bytes, dtype='<i2').astype(np.float32)
                raw_frame = raw_frame.reshape(NUM_CHANNELS, DISPLAY_BINS)
                # Firmware sends an overlapping display window, but advances
                # its source clock by only 5 or 6 fresh samples per frame.
                fresh_count = self._fresh_samples_for_frame(seq)
                self._append_raw_samples(raw_frame[:, :fresh_count])
            else:
                # FFT bins are unsigned magnitudes in microvolts.
                raw = np.frombuffer(data_bytes, dtype='<u2').astype(np.float32)
                raw *= self.calibration_scalar
                self.fft_data_matrix[:] = raw.reshape(NUM_CHANNELS, DISPLAY_BINS)
            self._frame_ready = True

            buf = buf[PROTO_FRAME_SIZE:]

        self._serial_buf = bytearray(buf)

    @staticmethod
    def _fresh_samples_for_frame(seq):
        """Match the firmware's 128-sample window / 20 ms source timing."""
        return 6 if seq % 25 in (0, 12, 24) else 5

    def _append_raw_samples(self, samples):
        """Append a (NUM_CHANNELS, N) block to the raw ring buffer."""
        n = int(samples.shape[1])
        if n <= 0:
            return

        if n >= RAW_WINDOW_BINS:
            samples = samples[:, -RAW_WINDOW_BINS:]
            n = RAW_WINDOW_BINS

        end = self._raw_ring_write + n
        if end <= RAW_WINDOW_BINS:
            self.raw_data_matrix[:, self._raw_ring_write:end] = samples
        else:
            first = RAW_WINDOW_BINS - self._raw_ring_write
            self.raw_data_matrix[:, self._raw_ring_write:] = samples[:, :first]
            self.raw_data_matrix[:, :end - RAW_WINDOW_BINS] = samples[:, first:]

        self._raw_ring_write = end % RAW_WINDOW_BINS
        self._raw_ring_count = min(RAW_WINDOW_BINS, self._raw_ring_count + n)
        self._raw_new_samples += n

    def _materialize_raw_view(self):
        """Copy the ring in chronological order for pyqtgraph rendering."""
        count = self._raw_ring_count
        if count <= 0:
            self._raw_render_matrix.fill(0.0)
            return self._raw_render_matrix

        if count < RAW_WINDOW_BINS:
            self._raw_render_matrix.fill(0.0)
            start = (self._raw_ring_write - count) % RAW_WINDOW_BINS
            if start + count <= RAW_WINDOW_BINS:
                self._raw_render_matrix[:, -count:] = self.raw_data_matrix[:, start:start + count]
            else:
                first = RAW_WINDOW_BINS - start
                self._raw_render_matrix[:, -count:-count + first] = self.raw_data_matrix[:, start:]
                remaining = count - first
                if remaining:
                    self._raw_render_matrix[:, -remaining:] = self.raw_data_matrix[:, :remaining]
            return self._raw_render_matrix

        start = self._raw_ring_write
        self._raw_indices[:] = (start + np.arange(RAW_WINDOW_BINS, dtype=np.int32)) % RAW_WINDOW_BINS
        np.take(self.raw_data_matrix, self._raw_indices, axis=1, out=self._raw_render_matrix)
        return self._raw_render_matrix

    @staticmethod
    def _crc16(data: bytes) -> int:
        """CRC-16/CCITT  poly=0x1021  init=0xFFFF — matches firmware."""
        crc = 0xFFFF
        for byte in data:
            crc = ((crc << 8) ^ CRC16_TABLE[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
        return crc

    def _poll_serial(self):
        if not self.ser or not self.ser.is_open:
            return

        try:
            waiting = self.ser.in_waiting
            if waiting > 0:
                self._serial_buf += self.ser.read(waiting)
                self._parse_serial_frames()
        except Exception:
            pass

    def update_plots(self):
        now = time.time()
        dt = now - self.last_update_time
        self.session_runtime  += dt
        self.last_update_time  = now

        total_cs    = int(self.session_runtime * 100)
        centiseconds = total_cs % 100
        total_s      = total_cs // 100
        self.runtime_label.setText(
            f"T+ {total_s // 60:02d}:{total_s % 60:02d}.{centiseconds:02d}"
        )

        if hasattr(self, "home_connection_value"):
            connected = bool(self.ser and self.ser.is_open)
            self.home_connection_value.setText("ONLINE" if connected else "OFFLINE")
            self.home_connection_value.setStyleSheet(
                f"color:{self._theme_color('#4EC9B0') if connected else '#F48771'}; font-size:22px; font-weight:bold;"
            )
            self.home_status_label.setText("●  DEVICE ONLINE" if connected else "●  SYSTEM READY")
            status_background = "#242424" if self.ui_theme == "Monochrome" else (
                "#183b3b" if connected else "#3b351d"
            )
            status_border = "#555555" if self.ui_theme == "Monochrome" else (
                "#245b59" if connected else "#665b2a"
            )
            self.home_status_label.setStyleSheet(
                f"color:{self._theme_color('#4EC9B0') if connected else self._theme_color('#dcdcaa')};"
                f" background:{status_background};"
                f" border:1px solid {status_border};"
                " border-radius:3px; padding:6px 10px; font-size:10px; font-weight:bold;"
            )
            self.home_connection_detail.setText(
                "Streaming from serial device" if connected else "Scan a port to begin"
            )
            active_channels = sum(cb.isChecked() for cb in self.checkboxes)
            self.home_channels_value.setText(f"{active_channels:02d} / {NUM_CHANNELS}")
            self.home_packets_value.setText(f"{self.packet_count:,}")
            self.home_packets_detail.setText(f"CRC errors: {self._crc_errors}")
            self.home_runtime_value.setText(f"{total_s // 3600:02d}:{(total_s % 3600) // 60:02d}:{total_s % 60:02d}")
            self.home_runtime_detail.setText(
                "Acquisition active" if connected else "Ready for acquisition"
            )

        if now - self.last_fps_time >= 1.0:
            stats_elapsed = max(0.001, now - self._last_throughput_time)
            packet_delta = self.packet_count - self._last_packet_count
            self._throughput_bps = packet_delta * PROTO_FRAME_SIZE * 8.0 / stats_elapsed
            self._last_packet_count = self.packet_count
            self._last_throughput_time = now
            if self.ser and self.ser.is_open:
                loss = (self.dropped_packets / max(1, self.packet_count)) * 100
                self.telemetry_label.setText(
                    f"Live | {packet_delta / stats_elapsed:.1f} pkt/s | Loss: {loss:.1f}% | CRC err: {self._crc_errors}"
                )
                color = "#E25C5C" if loss > 5 else self._theme_color("#00B37E")
                self.telemetry_label.setStyleSheet(
                    f"color:{color}; font-family:'Consolas'; font-size:12px; font-weight:bold;"
                )
            if hasattr(self, "home_throughput_value"):
                connected = bool(self.ser and self.ser.is_open)
                stream_active = connected and packet_delta > 0
                buffer_load = min(100.0, len(self._serial_buf) * 100.0 / max(1, PROTO_FRAME_SIZE * 2))
                self.home_throughput_value.setText(f"{self._throughput_bps / 1000.0:.1f} kbps")
                self.home_throughput_detail.setText(f"{packet_delta / stats_elapsed:.1f} packets/s")
                self.home_health_value.setText("STREAMING" if stream_active else "IDLE")
                self.home_health_value.setStyleSheet(
                    f"color:{self._theme_color('#68d6bd') if stream_active else self._theme_color('#d9b36c')}; font-size:25px; font-weight:bold;"
                )
                self.home_health_detail.setText(
                    f"Buffer load: {buffer_load:.0f}%  |  CRC errors: {self._crc_errors}"
                    if connected else "Connect hardware to begin"
                )
                self.home_event_label.setText(
                    "EVENTS  /  Stream healthy  /  No packet errors"
                    if stream_active and self._crc_errors == 0
                    else f"EVENTS  /  CRC errors: {self._crc_errors}  /  Dropped packets: {self.dropped_packets}"
                )
            self.last_fps_time = now

        if self.is_frozen:
            self._frame_ready = False
            self._raw_new_samples = 0
            return

        has_new_frame = self._frame_ready
        self._frame_ready = False

        if self.display_mode == "Raw Data" and self._raw_new_samples > 0:
            self._materialize_raw_view()
            self._raw_new_samples = 0
            has_new_frame = True

        if has_new_frame:
            for idx, curve in self.curves.items():
                curve.setData(
                    x=self._display_axis(),
                    y=self._plot_channel_values(idx),
                    skipFiniteCheck=True,
                )

            active_channels = list(self.curves)
            if active_channels and hasattr(self, "home_preview_curve"):
                preview_channel = active_channels[0]
                if self.display_mode == "Raw Data":
                    preview_data = self._materialize_raw_view()[preview_channel]
                    preview_axis = self.raw_x_axis
                    self.home_preview_mode.setText("RAW / CHANNEL 01")
                else:
                    preview_data = self.fft_data_matrix[preview_channel]
                    preview_axis = HZ_AXIS
                    self.home_preview_mode.setText("FFT / CHANNEL 01")
                self.home_preview_curve.setData(
                    x=preview_axis,
                    y=preview_data,
                    skipFiniteCheck=True,
                )

            if self.auto_scale_y:
                self._apply_auto_y_range()
        if hasattr(self, "graph_status_label"):
            source = "LIVE" if self.ser and self.ser.is_open else "PREVIEW"
            self.graph_status_label.setText(f"{source}  /  {len(self.curves):02d} TRACES")

        if has_new_frame and self._recording and self._csv_writer:
            record_matrix = self._plot_data_matrix()
            row = [f"{now:.4f}"] + [
                record_matrix[ch, b]
                for ch in self._record_channels
                for b in range(record_matrix.shape[1])
            ]
            self._csv_writer.writerow(row)
            if self._record_duration is not None:
                self._record_elapsed += dt
                remaining = max(0, self._record_duration - self._record_elapsed)
                self.record_btn.setText(f"⏹ {remaining:.0f}s left")
                if remaining <= 0:
                    self._stop_recording()

        self._peak_tick += 1
        if self._peak_tick >= PEAK_UPDATE_EVERY:
            self._peak_tick = 0
            indicator_matrix = self._plot_data_matrix()
            if self.is_split_view:
                for idx, wrapper in self.individual_widgets.items():
                    wrapper.update_peak(indicator_matrix[idx], self.display_mode)
                    wrapper.update_quality(indicator_matrix[idx])
                    self._update_peak_marker(idx, indicator_matrix[idx], wrapper.plot_widget)
            else:
                for idx in self.curves:
                    self._update_peak_marker(idx, indicator_matrix[idx], self.plot_widget)

    def closeEvent(self, event):
        if self._recording:
            self._stop_recording()
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        event.accept()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_UseOpenGLES)
    app    = QApplication(sys.argv)
    window = MultiChannelFFTApp()
    window.show()
    sys.exit(app.exec_())