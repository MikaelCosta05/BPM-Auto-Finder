import os
import re
import sys
import csv
import json
import wave
import random
import hashlib
import subprocess
import tempfile
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QUrl, QRectF, QTime
from PyQt6.QtGui import QAction, QLinearGradient, QColor, QBrush, QPen, QPainterPath, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget
)
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer, QSoundEffect

import pyqtgraph as pg


if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_env():
    env = os.environ.copy()
    possible_paths = [
        BASE_DIR,
        r"E:\msys64\ucrt64\bin",
        r"C:\msys64\ucrt64\bin"
    ]
    extra_paths = [path for path in possible_paths if os.path.isdir(path)]
    if extra_paths:
        env["PATH"] = ";".join(extra_paths) + ";" + env.get("PATH", "")
    return env


def find_ffmpeg():
    possible_paths = [
        os.path.join(BASE_DIR, "ffmpeg.exe"),
        r"E:\msys64\ucrt64\bin\ffmpeg.exe",
        r"C:\msys64\ucrt64\bin\ffmpeg.exe",
        "ffmpeg"
    ]
    for path in possible_paths:
        try:
            process = subprocess.run(
                [path, "-version"],
                text=True,
                capture_output=True,
                env=build_env()
            )
            if process.returncode == 0:
                return path
        except Exception:
            pass
    return None


def get_audio_duration(path):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    process = subprocess.run(
        [ffmpeg, "-i", path],
        text=True,
        capture_output=True,
        env=build_env()
    )
    output = process.stdout + process.stderr
    match = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", output)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def convert_to_temp_wav(source_path):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found.")
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp.name
    temp.close()
    process = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            source_path,
            "-ac",
            "1",
            "-ar",
            "44100",
            "-sample_fmt",
            "s16",
            temp_path
        ],
        text=True,
        capture_output=True,
        env=build_env()
    )
    if process.returncode != 0:
        try:
            os.remove(temp_path)
        except Exception:
            pass
        raise RuntimeError(process.stdout + process.stderr)
    return temp_path


def read_wav_mono(path):
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if sample_width != 2:
        raise RuntimeError("The WAV file must be 16-bit.")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if data.size:
        maximum = np.max(np.abs(data))
        if maximum > 0:
            data = data / maximum
    duration = len(data) / sample_rate
    return data, sample_rate, duration


def downsample_waveform(samples, sample_rate, max_points=35000):
    if len(samples) <= max_points:
        times = np.arange(len(samples)) / sample_rate
        return times, samples
    block = int(np.ceil(len(samples) / max_points))
    usable = (len(samples) // block) * block
    trimmed = samples[:usable]
    reshaped = trimmed.reshape(-1, block)
    reduced = np.max(np.abs(reshaped), axis=1)
    signs = np.sign(np.mean(reshaped, axis=1))
    reduced = reduced * np.where(signs == 0, 1, signs)
    times = np.arange(len(reduced)) * block / sample_rate
    return times, reduced


def moving_average(values, size):
    if size <= 1:
        return values
    kernel = np.ones(size) / size
    return np.convolve(values, kernel, mode="same")


def median_filter(values, size):
    if size <= 1:
        return values
    radius = size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    result = np.zeros_like(values)
    for index in range(len(values)):
        result[index] = np.median(padded[index:index + size])
    return result


def refine_kick_time(samples, sample_rate, time_value):
    center = int(time_value * sample_rate)
    radius = int(0.040 * sample_rate)
    start = max(0, center - radius)
    end = min(len(samples), center + radius)
    if end <= start:
        return time_value
    segment = np.abs(samples[start:end])
    if len(segment) < 2:
        return time_value
    diff = np.diff(segment)
    peak = int(np.argmax(diff))
    return (start + peak) / sample_rate


def detect_kicks_range(samples, sample_rate, range_start, range_end, min_gap=0.28, sensitivity=1.75):
    start_sample = max(0, int(range_start * sample_rate))
    end_sample = min(len(samples), int(range_end * sample_rate))
    segment = samples[start_sample:end_sample]
    kicks = detect_kicks(segment, sample_rate, min_gap=min_gap, sensitivity=sensitivity)
    return [(time_value + range_start, strength) for time_value, strength in kicks]


def detect_kicks(samples, sample_rate, min_gap=0.28, sensitivity=1.75):
    frame_size = 4096
    hop_size = 256
    if len(samples) < frame_size:
        return []

    window = np.hanning(frame_size)
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
    sub_mask = (frequencies >= 35) & (frequencies <= 95)
    kick_mask = (frequencies >= 35) & (frequencies <= 155)
    low_mask = (frequencies >= 20) & (frequencies <= 190)
    mid_mask = (frequencies >= 220) & (frequencies <= 1800)
    high_mask = (frequencies >= 2500) & (frequencies <= 9000)

    energies = []
    times = []
    previous_low = 0.0

    for start in range(0, len(samples) - frame_size, hop_size):
        frame = samples[start:start + frame_size] * window
        spectrum = np.abs(np.fft.rfft(frame))
        power = spectrum * spectrum

        sub_energy = np.sum(power[sub_mask])
        kick_energy = np.sum(power[kick_mask])
        low_energy = np.sum(power[low_mask])
        mid_energy = np.sum(power[mid_mask]) + 1e-9
        high_energy = np.sum(power[high_mask]) + 1e-9

        attack = max(0.0, low_energy - previous_low)
        previous_low = low_energy

        low_ratio = low_energy / (low_energy + mid_energy + high_energy)
        sub_ratio = sub_energy / (kick_energy + 1e-9)
        mid_penalty = 1.0 / (1.0 + mid_energy / (low_energy + 1e-9))
        high_penalty = 1.0 / (1.0 + high_energy / (low_energy + 1e-9))

        score = attack * low_ratio * (0.70 + sub_ratio) * (0.65 + mid_penalty) * (0.75 + high_penalty)

        energies.append(score)
        times.append((start + frame_size / 2) / sample_rate)

    energies = np.array(energies, dtype=np.float32)
    times = np.array(times, dtype=np.float32)

    if len(energies) < 12:
        return []

    smooth = moving_average(energies, 3)
    baseline = median_filter(smooth, 41)
    novelty = smooth - baseline
    novelty[novelty < 0] = 0
    novelty = moving_average(novelty, 3)

    maximum = np.max(novelty)
    if maximum <= 0:
        return []

    threshold = np.median(novelty) + np.std(novelty) * sensitivity
    threshold = max(threshold, maximum * 0.08)

    candidates = []

    for i in range(2, len(novelty) - 2):
        if novelty[i] > threshold and novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1] and novelty[i] >= novelty[i - 2] and novelty[i] >= novelty[i + 2]:
            refined_time = refine_kick_time(samples, sample_rate, float(times[i]))
            candidates.append((refined_time, float(novelty[i])))

    selected = []

    for time_value, strength in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(time_value - existing[0]) >= min_gap for existing in selected):
            selected.append((time_value, strength))

    selected.sort(key=lambda item: item[0])
    return selected


def filter_primary_kicks(kicks, bpm, tolerance=0.060):
    if not kicks or bpm <= 0:
        return kicks

    interval = 60.0 / bpm
    if interval <= 0:
        return kicks

    strengths = np.array([kick[1] for kick in kicks], dtype=np.float32)
    if len(strengths) == 0:
        return kicks

    strong_limit = np.percentile(strengths, 45)
    strong_kicks = [kick for kick in kicks if kick[1] >= strong_limit]
    if not strong_kicks:
        strong_kicks = kicks

    best_score = None
    best_offset = strong_kicks[0][0] % interval

    for kick_time, kick_strength in strong_kicks[:120]:
        offset = kick_time % interval
        score = 0.0
        for candidate_time, candidate_strength in kicks:
            phase = abs(((candidate_time - offset + interval / 2) % interval) - interval / 2)
            if phase <= tolerance:
                score += candidate_strength * (1.0 - phase / tolerance)
        if best_score is None or score > best_score or (score == best_score and error < best[3]):
            best_score = score
            best_offset = offset

    primary = []

    for kick_time, kick_strength in kicks:
        phase = abs(((kick_time - best_offset + interval / 2) % interval) - interval / 2)
        half_phase = abs(((kick_time - best_offset - interval / 2 + interval / 2) % interval) - interval / 2)
        if phase <= tolerance or half_phase <= tolerance:
            primary.append((kick_time, kick_strength))

    return primary or kicks


def fit_bpm_offset_to_kicks(kicks, initial_bpm):
    if not kicks:
        return initial_bpm, 0.0, 0.0

    kick_times = np.array([kick[0] for kick in kicks], dtype=np.float64)
    strengths = np.array([kick[1] for kick in kicks], dtype=np.float64)
    if len(kick_times) < 2:
        return initial_bpm, float(kick_times[0] if len(kick_times) else 0.0), 0.0

    strengths = strengths / (np.max(strengths) + 1e-9)
    best = None

    if initial_bpm > 0:
        bpm_range = np.linspace(initial_bpm * 0.965, initial_bpm * 1.035, 121)
    else:
        bpm_range = np.linspace(70.0, 180.0, 1101)

    for bpm in bpm_range:
        interval = 60.0 / bpm
        for reference in kick_times[:min(len(kick_times), 40)]:
            offset = reference % interval
            distances = np.abs(((kick_times - offset + interval / 2) % interval) - interval / 2)
            closeness = np.maximum(0.0, 1.0 - distances / min(0.140, interval * 0.35))
            score = float(np.sum(closeness * strengths))
            error = float(np.average(distances, weights=strengths + 1e-6))
            if best is None or score > best[0] or (score == best[0] and error < best[3]):
                best = (score, bpm, offset, error)

    if best is None:
        return initial_bpm, float(kick_times[0] % (60.0 / initial_bpm)), 0.0

    return float(best[1]), float(best[2]), float(best[0])


def detect_toms(samples, sample_rate, min_gap=0.20, sensitivity=1.6):
    frame_size = 4096
    hop_size = 256
    if len(samples) < frame_size:
        return []

    window = np.hanning(frame_size)
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
    tom_mask = (frequencies >= 80) & (frequencies <= 300)
    attack_mask = (frequencies >= 500) & (frequencies <= 3000)

    energies = []
    times = []
    previous_tom = 0.0

    for start in range(0, len(samples) - frame_size, hop_size):
        frame = samples[start:start + frame_size] * window
        spectrum = np.abs(np.fft.rfft(frame))
        power = spectrum * spectrum

        tom_energy = np.sum(power[tom_mask])
        attack_energy = np.sum(power[attack_mask]) + 1e-9

        attack = max(0.0, tom_energy - previous_tom)
        previous_tom = tom_energy

        score = attack * (0.8 + attack_energy / (tom_energy + 1e-9))
        energies.append(score)
        times.append((start + frame_size / 2) / sample_rate)

    energies = np.array(energies, dtype=np.float32)
    times = np.array(times, dtype=np.float32)

    if len(energies) < 12:
        return []

    smooth = moving_average(energies, 3)
    baseline = median_filter(smooth, 41)
    novelty = smooth - baseline
    novelty[novelty < 0] = 0
    novelty = moving_average(novelty, 3)

    maximum = np.max(novelty)
    if maximum <= 0:
        return []

    threshold = np.median(novelty) + np.std(novelty) * sensitivity
    threshold = max(threshold, maximum * 0.08)

    candidates = []

    for i in range(2, len(novelty) - 2):
        if novelty[i] > threshold and novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1] and novelty[i] >= novelty[i - 2] and novelty[i] >= novelty[i + 2]:
            refined_time = refine_kick_time(samples, sample_rate, float(times[i]))
            candidates.append((refined_time, float(novelty[i])))

    selected = []

    for time_value, strength in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(time_value - existing[0]) >= min_gap for existing in selected):
            selected.append((time_value, strength))

    selected.sort(key=lambda item: item[0])
    return selected


class GradientLineItem(pg.GraphicsObject):
    def __init__(self, points_count=256):
        super().__init__()
        self.points_count = points_count
        self.x = np.arange(points_count)
        self.y = np.zeros(points_count)
        self.gradient = QLinearGradient(0, 0, points_count, 0)
        self.gradient.setColorAt(0.0, QColor('#ffd700'))
        self.gradient.setColorAt(0.2, QColor('#ff4500'))
        self.gradient.setColorAt(0.4, QColor('#e60067'))
        self.gradient.setColorAt(0.6, QColor('#8a2be2'))
        self.gradient.setColorAt(0.8, QColor('#1e90ff'))
        self.gradient.setColorAt(1.0, QColor('#00ffff'))
        self.pen = QPen(QBrush(self.gradient), 3)
        self.pen.setCosmetic(True)
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

    def setData(self, y):
        self.y = y
        self.update()

    def boundingRect(self):
        return QRectF(0, -1.5, float(self.points_count), 3.0)

    def paint(self, painter, option, widget):
        if len(self.y) == 0:
            return
        path = QPainterPath()
        path.moveTo(float(self.x[0]), float(self.y[0]))
        for i in range(1, len(self.x)):
            path.lineTo(float(self.x[i]), float(self.y[i]))
        painter.setPen(self.pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)


def write_varlen(value):
    buffer = bytearray()
    buffer.append(value & 0x7F)
    value >>= 7
    while value > 0:
        buffer.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(buffer))


def create_midi_file(filepath, bpm, events_data, default_pitch=36):
    tpqn = 480
    bpm = max(bpm, 1.0)
    us_per_qn = int(round(60000000.0 / bpm))
    raw_events = []
    for item in events_data:
        if len(item) == 3:
            t, vel, pitch = item
        else:
            t, vel = item
            pitch = default_pitch
        t_on = max(0.0, float(t))
        t_off = t_on + 0.1
        v_int = int(min(127, max(1, vel * 127)))
        raw_events.append((t_on, 1, pitch, v_int))
        raw_events.append((t_off, 0, pitch, 0))
    raw_events.sort(key=lambda x: (x[0], x[1]))
    track_data = bytearray()
    track_data.extend(b'\x00\xFF\x51\x03')
    track_data.append((us_per_qn >> 16) & 0xFF)
    track_data.append((us_per_qn >> 8) & 0xFF)
    track_data.append(us_per_qn & 0xFF)
    last_ticks = 0
    for t_sec, ev_type, pitch, vel in raw_events:
        ticks = int(round((t_sec * 1000000.0 / us_per_qn) * tpqn))
        delta_ticks = max(0, ticks - last_ticks)
        last_ticks = ticks
        track_data.extend(write_varlen(delta_ticks))
        if ev_type == 1:
            track_data.extend(bytes([0x99, pitch, vel]))
        else:
            track_data.extend(bytes([0x89, pitch, 0]))
    track_data.extend(b'\x00\xFF\x2F\x00')
    header = b'MThd\x00\x00\x00\x06\x00\x00\x00\x01' + tpqn.to_bytes(2, 'big')
    track = b'MTrk' + len(track_data).to_bytes(4, 'big') + bytes(track_data)
    with open(filepath, 'wb') as f:
        f.write(header + track)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BPM Offset Detector Visual")
        self.resize(1360, 820)

        icon_path = os.path.join(BASE_DIR, "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.audio_path = ""
        self.temp_wav_path = ""
        self.duration = 0.0
        self.raw_samples = np.array([])
        self.sample_rate = 44100
        self.wave_times = np.array([])
        self.wave_values = np.array([])
        self.beat_lines = []
        self.kick_lines = []
        self.segment_lines = []
        self.detected_segments = []
        self.tom_lines = []
        self.kicks = []
        self.toms = []
        self.all_kicks = []
        self.kick_strength_curve = None

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)

        self.kick_path = self.create_drum_file("bpm_kick.wav", 400, 50, 0.25, 1.0, 30.0, 10.0)
        self.tom_path = self.create_drum_file("bpm_tom.wav", 250, 100, 0.25, 0.8, 25.0, 12.0)
        self.kick_sound = QSoundEffect()
        self.tom_sound = QSoundEffect()
        self.kick_sound.setSource(QUrl.fromLocalFile(self.kick_path))
        self.tom_sound.setSource(QUrl.fromLocalFile(self.tom_path))
        self.kick_sound.setVolume(0.9)
        self.tom_sound.setVolume(0.9)

        self.metronome_bpm = 0.0
        self.metronome_offset = 0.0
        self.next_beat_time = None
        self.next_kick_index = 0
        self.next_tom_index = 0
        self.metronome_mode = "kicks"

        self.spectrum_points = 256
        self.spectrum_x = np.arange(self.spectrum_points)
        self.metronome_boost = 0.0

        self.timer = QTimer()
        self.timer.setInterval(12)
        self.timer.timeout.connect(self.update_cursor)

        self.setup_ui()
        self.setup_menu()
        self.apply_theme()

    def create_drum_file(self, filename, start_freq, end_freq, duration, volume, pitch_decay, amp_decay):
        path = os.path.join(tempfile.gettempdir(), filename)
        sample_rate = 44100
        samples_count = int(sample_rate * duration)
        t = np.arange(samples_count) / sample_rate
        freqs = end_freq + (start_freq - end_freq) * np.exp(-t * pitch_decay)
        phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
        envelope = np.exp(-t * amp_decay)
        samples = np.clip(np.sin(phase) * envelope * volume * 2.0, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(samples.tobytes())
        return path

    def setup_menu(self):
        menu = self.menuBar().addMenu("File")
        open_action = QAction("Open Song", self)
        open_action.triggered.connect(self.choose_file)
        menu.addAction(open_action)

        export_csv_action = QAction("Export CSV", self)
        export_csv_action.triggered.connect(self.export_csv)
        menu.addAction(export_csv_action)

        export_json_action = QAction("Export JSON", self)
        export_json_action.triggered.connect(self.export_json)
        menu.addAction(export_json_action)

        export_pagoda_action = QAction("Export to Pagoda", self)
        export_pagoda_action.triggered.connect(self.export_pagoda)
        menu.addAction(export_pagoda_action)

        export_midi_action = QAction("Export MIDI", self)
        export_midi_action.triggered.connect(self.export_midi)
        menu.addAction(export_midi_action)

    def log_text(self, text):
        now = QTime.currentTime().toString("hh:mm:ss")
        if text.startswith("---") or text == "":
            self.log.appendPlainText(text)
        else:
            self.log.appendPlainText(f"[{now}] {text}")

    def apply_theme(self):
        style = """
            QMainWindow { background-color: #0c0e14; }
            QWidget { color: #94a3b8; font-family: 'Segoe UI', sans-serif; font-size: 12px; }
            QFrame#sidebar { background-color: #11141d; border-right: 1px solid #1e2436; }
            QFrame#card { background-color: #181c27; border: 1px solid #23283b; border-radius: 8px; }
            QFrame#header_card { background-color: #181c27; border: 1px solid #23283b; border-radius: 8px; }
            QLabel { color: #cbd5e1; font-weight: 500; }
            
            QPushButton { background-color: #1e2436; color: #f1f5f9; border: 1px solid #2d354d; border-radius: 6px; padding: 6px 12px; font-weight: 600; }
            QPushButton:hover { background-color: #2a324b; border-color: #3b4666; }
            QPushButton:pressed { background-color: #1a2030; }
            
            QPushButton#btn_primary { background-color: #7c5cfc; color: #ffffff; border: none; }
            QPushButton#btn_primary:hover { background-color: #6d4aff; }
            
            QPushButton#btn_success { background-color: #10b981; color: #ffffff; border: none; }
            QPushButton#btn_success:hover { background-color: #059669; }
            
            QPushButton#btn_danger { background-color: #e11d48; color: #ffffff; border: none; }
            QPushButton#btn_danger:hover { background-color: #be123c; }

            QPushButton#btn_outline_green { background-color: #122822; color: #34d399; border: 1px solid #059669; }
            QPushButton#btn_outline_green:hover { background-color: #1b3d33; }

            QPushButton#btn_outline_blue { background-color: #132338; color: #60a5fa; border: 1px solid #2563eb; }
            QPushButton#btn_outline_blue:hover { background-color: #1d3654; }

            QPushButton#btn_outline_orange { background-color: #2b1f15; color: #fbbf24; border: 1px solid #d97706; }
            QPushButton#btn_outline_orange:hover { background-color: #402e1c; }

            QLineEdit { background-color: #0f121d; color: #34d399; border: 1px solid #23283b; border-radius: 6px; padding: 5px; font-weight: bold; font-family: 'Consolas', monospace; }
            QLineEdit:focus { border: 1px solid #7c5cfc; }
            
            QComboBox { background-color: #181c27; color: #f1f5f9; border: 1px solid #23283b; border-radius: 6px; padding: 4px 8px; }
            QCheckBox { color: #cbd5e1; font-weight: 500; }
            
            QPlainTextEdit { background-color: #0c0e14; color: #34d399; border: 1px solid #1e2436; border-radius: 6px; font-family: 'Consolas', monospace; font-size: 11px; }
            QSplitter::handle { background-color: #1e2436; }
            QStatusBar { background-color: #11141d; color: #64748b; border-top: 1px solid #1e2436; font-size: 11px; }
            
            QSlider::groove:horizontal { height: 4px; background: #23283b; border-radius: 2px; }
            QSlider::handle:horizontal { background: #7c5cfc; width: 12px; margin: -4px 0; border-radius: 6px; }
            QSlider::sub-page:horizontal { background: #7c5cfc; border-radius: 2px; }
        """
        self.setStyleSheet(style)

    def clear_all_markers(self):
        self.clear_beat_lines()
        self.clear_kick_lines()
        self.clear_tom_lines()
        self.clear_segment_lines()
        self.kicks = []
        self.toms = []
        self.all_kicks = []
        self.detected_segments = []
        self.bpm_input.clear()
        self.offset_input.clear()
        self.kick_curve.setData([], [])
        self.spectrum_curve.setData(np.zeros(256))
        self.log_text("All markers, kicks, toms, and BPMs have been cleared.")

    def setup_ui(self):
        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setSpacing(0)
        root_layout.setContentsMargins(0, 0, 0, 0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 16, 12, 16)
        sidebar_layout.setSpacing(12)

        logo_label = QLabel("BPM Offset Detector")
        logo_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f8fafc;")
        sidebar_layout.addWidget(logo_label)

        lbl_sec1 = QLabel("FILE")
        lbl_sec1.setStyleSheet("font-size: 10px; font-weight: 700; color: #475569; margin-top: 10px;")
        sidebar_layout.addWidget(lbl_sec1)

        self.open_button = QPushButton("📁 Open Song")
        self.open_button.clicked.connect(self.choose_file)
        sidebar_layout.addWidget(self.open_button)

        lbl_sec2 = QLabel("METRONOME")
        lbl_sec2.setStyleSheet("font-size: 10px; font-weight: 700; color: #475569; margin-top: 10px;")
        sidebar_layout.addWidget(lbl_sec2)

        self.kick_metronome_button = QPushButton("Metronome: BPM")
        self.kick_metronome_button.clicked.connect(self.toggle_metronome_mode)
        sidebar_layout.addWidget(self.kick_metronome_button)

        vol_box = QHBoxLayout()
        vol_box.addWidget(QLabel("Vol"))
        self.click_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.click_volume_slider.setRange(0, 200)
        self.click_volume_slider.setValue(100)
        self.click_volume_slider.valueChanged.connect(self.update_click_volume)
        vol_box.addWidget(self.click_volume_slider)
        sidebar_layout.addLayout(vol_box)

        lbl_sec3 = QLabel("ANALYSIS SECTION")
        lbl_sec3.setStyleSheet("font-size: 10px; font-weight: 700; color: #475569; margin-top: 10px;")
        sidebar_layout.addWidget(lbl_sec3)

        p1 = QHBoxLayout()
        p1.addWidget(QLabel("Section (s)"))
        self.segment_input = QLineEdit("30")
        p1.addWidget(self.segment_input)
        sidebar_layout.addLayout(p1)

        p2 = QHBoxLayout()
        p2.addWidget(QLabel("Hop"))
        self.hop_input = QLineEdit("256")
        p2.addWidget(self.hop_input)
        sidebar_layout.addLayout(p2)

        p3 = QHBoxLayout()
        p3.addWidget(QLabel("Sensitivity"))
        self.sensitivity_input = QLineEdit("1.75")
        p3.addWidget(self.sensitivity_input)
        sidebar_layout.addLayout(p3)

        p4 = QHBoxLayout()
        p4.addWidget(QLabel("Distance (s)"))
        self.kick_gap_input = QLineEdit("0.28")
        p4.addWidget(self.kick_gap_input)
        sidebar_layout.addLayout(p4)

        sens_btn_box = QHBoxLayout()
        self.sens_minus_button = QPushButton("Sens -")
        self.sens_minus_button.clicked.connect(lambda: self.adjust_sensitivity(-0.10))
        self.sens_plus_button = QPushButton("Sens +")
        self.sens_plus_button.clicked.connect(lambda: self.adjust_sensitivity(0.10))
        sens_btn_box.addWidget(self.sens_minus_button)
        sens_btn_box.addWidget(self.sens_plus_button)
        sidebar_layout.addLayout(sens_btn_box)

        gap_btn_box = QHBoxLayout()
        self.gap_minus_button = QPushButton("Dist -")
        self.gap_minus_button.clicked.connect(lambda: self.adjust_gap(-0.02))
        self.gap_plus_button = QPushButton("Dist +")
        self.gap_plus_button.clicked.connect(lambda: self.adjust_gap(0.02))
        gap_btn_box.addWidget(self.gap_minus_button)
        gap_btn_box.addWidget(self.gap_plus_button)
        sidebar_layout.addLayout(gap_btn_box)

        self.adaptive_check = QCheckBox("Section-based analysis")
        self.adaptive_check.setChecked(True)
        sidebar_layout.addWidget(self.adaptive_check)

        sidebar_layout.addStretch()
        root_layout.addWidget(sidebar)

        main_content = QWidget()
        main_layout = QVBoxLayout(main_content)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        header_card = QFrame()
        header_card.setObjectName("header_card")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(12, 8, 12, 8)

        self.file_label = QLabel("No song selected")
        self.file_label.setStyleSheet("font-weight: bold; color: #f8fafc;")
        header_layout.addWidget(self.file_label)
        header_layout.addStretch()

        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("btn_primary")
        self.play_button.clicked.connect(self.toggle_play)

        self.stop_button = QPushButton("Reset")
        self.stop_button.clicked.connect(self.stop_audio)

        header_layout.addWidget(self.play_button)
        header_layout.addWidget(self.stop_button)

        main_layout.addWidget(header_card)

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)

        card_an = QFrame()
        card_an.setObjectName("card")
        l_an = QVBoxLayout(card_an)
        lbl_an = QLabel("DETECTION & ANALYSIS")
        lbl_an.setStyleSheet("font-size: 10px; font-weight: bold; color: #818cf8;")
        l_an.addWidget(lbl_an)

        r_an1 = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze BPM")
        self.analyze_button.setObjectName("btn_primary")
        self.analyze_button.clicked.connect(self.analyze_full)
        self.segments_button = QPushButton("Multiple BPMs")
        self.segments_button.clicked.connect(self.analyze_segments)
        self.auto_fit_button = QPushButton("Fit BPM/Offset")
        self.auto_fit_button.clicked.connect(self.auto_fit_bpm_offset)
        r_an1.addWidget(self.analyze_button)
        r_an1.addWidget(self.segments_button)
        r_an1.addWidget(self.auto_fit_button)
        l_an.addLayout(r_an1)

        r_an2 = QHBoxLayout()
        self.kicks_button = QPushButton("Detect Kicks")
        self.kicks_button.setObjectName("btn_outline_green")
        self.kicks_button.clicked.connect(self.detect_and_draw_kicks)
        self.primary_kicks_button = QPushButton("Primary Kick")
        self.primary_kicks_button.setObjectName("btn_outline_blue")
        self.primary_kicks_button.clicked.connect(self.apply_primary_kick_mode)
        self.toms_button = QPushButton("Detect Toms")
        self.toms_button.setObjectName("btn_outline_orange")
        self.toms_button.clicked.connect(self.detect_and_draw_toms)
        r_an2.addWidget(self.kicks_button)
        r_an2.addWidget(self.primary_kicks_button)
        r_an2.addWidget(self.toms_button)
        l_an.addLayout(r_an2)
        cards_layout.addWidget(card_an)

        card_grid = QFrame()
        card_grid.setObjectName("card")
        l_grid = QVBoxLayout(card_grid)
        lbl_grid = QLabel("GRID & OFFSET")
        lbl_grid.setStyleSheet("font-size: 10px; font-weight: bold; color: #34d399;")
        l_grid.addWidget(lbl_grid)

        r_g1 = QHBoxLayout()
        r_g1.addWidget(QLabel("BPM:"))
        self.bpm_input = QLineEdit()
        self.bpm_input.setFixedWidth(85)
        r_g1.addWidget(self.bpm_input)

        r_g1.addWidget(QLabel("Offset (ms):"))
        self.offset_input = QLineEdit()
        self.offset_input.setFixedWidth(75)
        r_g1.addWidget(self.offset_input)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("btn_success")
        self.apply_button.clicked.connect(self.apply_beats)
        r_g1.addWidget(self.apply_button)
        l_grid.addLayout(r_g1)

        r_g2 = QHBoxLayout()
        self.offset_minus_big_button = QPushButton("-50ms")
        self.offset_minus_big_button.clicked.connect(lambda: self.adjust_offset(-0.050))
        self.offset_minus_button = QPushButton("-10ms")
        self.offset_minus_button.clicked.connect(lambda: self.adjust_offset(-0.010))
        self.offset_plus_button = QPushButton("+10ms")
        self.offset_plus_button.clicked.connect(lambda: self.adjust_offset(0.010))
        self.offset_plus_big_button = QPushButton("+50ms")
        self.offset_plus_big_button.clicked.connect(lambda: self.adjust_offset(0.050))
        r_g2.addWidget(self.offset_minus_big_button)
        r_g2.addWidget(self.offset_minus_button)
        r_g2.addWidget(self.offset_plus_button)
        r_g2.addWidget(self.offset_plus_big_button)
        l_grid.addLayout(r_g2)
        cards_layout.addWidget(card_grid)

        card_exp = QFrame()
        card_exp.setObjectName("card")
        l_exp = QVBoxLayout(card_exp)
        lbl_exp = QLabel("ACTIONS & EXPORT")
        lbl_exp.setStyleSheet("font-size: 10px; font-weight: bold; color: #c084fc;")
        l_exp.addWidget(lbl_exp)

        r_e1 = QHBoxLayout()
        self.export_pagoda_button = QPushButton("Export Pagoda")
        self.export_pagoda_button.setObjectName("btn_primary")
        self.export_pagoda_button.clicked.connect(self.export_pagoda)
        self.export_midi_button = QPushButton("Export MIDI")
        self.export_midi_button.clicked.connect(self.export_midi)
        r_e1.addWidget(self.export_pagoda_button)
        r_e1.addWidget(self.export_midi_button)
        l_exp.addLayout(r_e1)

        r_e2 = QHBoxLayout()
        self.clear_all_button = QPushButton("Clear Markers")
        self.clear_all_button.setObjectName("btn_danger")
        self.clear_all_button.clicked.connect(self.clear_all_markers)
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["Full", "10s", "20s", "30s", "60s", "120s"])
        self.zoom_combo.currentTextChanged.connect(self.apply_zoom)
        r_e2.addWidget(self.clear_all_button)
        r_e2.addWidget(QLabel("Zoom:"))
        r_e2.addWidget(self.zoom_combo)
        l_exp.addLayout(r_e2)
        cards_layout.addWidget(card_exp)

        main_layout.addLayout(cards_layout)

        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setBackground("#0c0e14")
        self.wave_plot.showGrid(x=True, y=False, alpha=0.15)
        self.wave_plot.setMouseEnabled(x=True, y=False)
        self.wave_plot.setMenuEnabled(False)
        self.wave_plot.setLabel("bottom", "Time", units="s")
        self.wave_plot.setYRange(-1.05, 1.05)
        self.wave_plot.hideAxis("left")

        self.wave_item = self.wave_plot.plot([], [], pen=pg.mkPen("#a855f7", width=1))
        self.cursor_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#ffffff", width=2))
        self.wave_plot.addItem(self.cursor_line)
        self.wave_plot.scene().sigMouseClicked.connect(self.seek_to_plot_click)

        self.kick_plot = pg.PlotWidget()
        self.kick_plot.setBackground("#0c0e14")
        self.kick_plot.showGrid(x=True, y=True, alpha=0.15)
        self.kick_plot.setMouseEnabled(x=True, y=False)
        self.kick_plot.setMenuEnabled(False)
        self.kick_plot.setLabel("bottom", "Kick Strength")
        self.kick_plot.setYRange(0, 1.05)
        self.kick_plot.hideAxis("left")
        self.kick_curve = self.kick_plot.plot([], [], pen=pg.mkPen("#34d399", width=1))

        self.wave_plot.setXLink(self.kick_plot)

        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground("#0c0e14")
        self.spectrum_plot.setYRange(-1.2, 1.2)
        self.spectrum_plot.hideAxis('left')
        self.spectrum_plot.hideAxis('bottom')
        self.spectrum_plot.setMouseEnabled(x=False, y=False)
        self.spectrum_plot.setMenuEnabled(False)

        self.spectrum_curve = GradientLineItem(256)
        self.spectrum_plot.addItem(self.spectrum_curve)

        splitter = QSplitter(Qt.Orientation.Vertical)
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.wave_plot, 5)
        plot_layout.addWidget(self.kick_plot, 1)
        plot_layout.addWidget(self.spectrum_plot, 1)
        splitter.addWidget(plot_container)

        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        lbl_console = QLabel("CONSOLE")
        lbl_console.setStyleSheet("font-size: 10px; font-weight: bold; color: #818cf8;")
        bottom_layout.addWidget(lbl_console)

        self.position_label = QLabel("00:00 / 00:00")
        bottom_layout.addWidget(self.position_label)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        bottom_layout.addWidget(self.log)

        splitter.addWidget(bottom_container)
        splitter.setSizes([560, 220])

        main_layout.addWidget(splitter)
        root_layout.addWidget(main_content)

        self.statusBar().showMessage("Ready")
        self.setCentralWidget(central)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Song",
            "",
            "Audio (*.mp3 *.wav *.flac *.ogg *.m4a);;All files (*.*)"
        )
        if not path:
            return
        self.load_audio(path)

    def load_audio(self, path):
        self.stop_audio()
        self.clear_beat_lines()
        self.clear_kick_lines()
        self.clear_tom_lines()
        self.clear_segment_lines()
        self.detected_segments = []
        self.audio_path = path
        self.file_label.setText(path)
        self.log.clear()
        self.log_text("Loading waveform...")
        try:
            if self.temp_wav_path and os.path.isfile(self.temp_wav_path):
                try:
                    os.remove(self.temp_wav_path)
                except Exception:
                    pass
            self.temp_wav_path = convert_to_temp_wav(path)
            samples, sample_rate, duration = read_wav_mono(self.temp_wav_path)
            self.raw_samples = samples
            self.sample_rate = sample_rate
            self.duration = duration
            self.kicks = []
            self.toms = []
            self.all_kicks = []
            self.next_kick_index = 0
            self.next_tom_index = 0
            self.wave_times, self.wave_values = downsample_waveform(samples, sample_rate)
            self.wave_item.setData(self.wave_times, self.wave_values)
            self.kick_curve.setData([], [])
            self.wave_plot.setXRange(0, max(self.duration, 1), padding=0)
            self.player.setSource(QUrl.fromLocalFile(path))
            self.update_position_label(0)
            self.log_text(f"Song loaded: {os.path.basename(path)}")
            self.log_text(f"Duration: {self.duration:.2f}s")
        except Exception as error:
            QMessageBox.critical(self, "Error", str(error))

    def validate_audio(self):
        if not self.audio_path:
            QMessageBox.warning(self, "Warning", "Choose a song first.")
            return False
        if not os.path.isfile(self.audio_path):
            QMessageBox.critical(self, "Error", "The selected file does not exist.")
            return False
        return True

    def analyze_full(self):
        if not self.validate_audio():
            return
        self.centralWidget().setEnabled(False)
        try:
            self.log.clear()
            self.clear_segment_lines()
            self.detected_segments = []
            self.log_text("Analyzing BPM of the full song (native mode)...")
            QApplication.processEvents()
            
            try:
                sens = float(self.sensitivity_input.text().strip())
                gap = float(self.kick_gap_input.text().strip())
            except Exception:
                sens = 1.75
                gap = 0.28
            
            if self.adaptive_check.isChecked():
                try:
                    segment_size = float(self.segment_input.text().strip())
                except Exception:
                    segment_size = 30.0
                kicks_all = []
                start = 0.0
                while start < self.duration:
                    end = min(self.duration, start + segment_size)
                    kicks_all.extend(detect_kicks_range(self.raw_samples, self.sample_rate, start, end, min_gap=gap, sensitivity=sens))
                    start += segment_size
            else:
                kicks_all = detect_kicks(self.raw_samples, self.sample_rate, min_gap=gap, sensitivity=sens)
            
            if len(kicks_all) >= 4:
                bpm, offset, score = fit_bpm_offset_to_kicks(kicks_all, 0.0)
                if bpm > 0:
                    self.bpm_input.setText(f"{bpm:.6f}")
                    self.offset_input.setText(f"{offset * 1000.0:.2f}")
                    self.log_text("")
                    self.log_text(f"BPM: {bpm:.6f}")
                    self.log_text(f"Offset: {offset * 1000.0:.2f}ms")
                    self.log_text(f"Fitness: {score * 50.0:.6f}")
                    self.draw_beats(bpm, offset)
                    return

            QMessageBox.warning(self, "Warning", "Could not detect BPM/offset in this file.")
        finally:
            self.centralWidget().setEnabled(True)

    def analyze_segments(self):
        if not self.validate_audio():
            return
        try:
            segment_size = float(self.segment_input.text().strip())
            if segment_size <= 0:
                raise ValueError()
        except Exception:
            QMessageBox.critical(self, "Error", "The section size must be greater than zero.")
            return

        self.centralWidget().setEnabled(False)
        try:
            self.log.clear()
            self.clear_segment_lines()
            self.detected_segments = []
            self.log_text("Detecting multiple BPMs (native analysis)...")
            QApplication.processEvents()

            step = segment_size / 2.0
            start = 0.0
            raw_results = []

            try:
                sens = float(self.sensitivity_input.text().strip())
                gap = float(self.kick_gap_input.text().strip())
            except Exception:
                sens = 1.75
                gap = 0.28

            while start < self.duration:
                current_duration = min(segment_size, self.duration - start)
                if current_duration < segment_size * 0.33 and raw_results:
                    break
                
                kicks_in_range = detect_kicks_range(self.raw_samples, self.sample_rate, start, start + current_duration, min_gap=gap, sensitivity=sens)
                if len(kicks_in_range) >= 4:
                    fb_bpm, fb_offset, fb_score = fit_bpm_offset_to_kicks(kicks_in_range, 0.0)
                    if fb_bpm > 0:
                        result = {
                            "bpm": fb_bpm,
                            "offset": fb_offset,
                            "fitness": fb_score * 50.0,
                            "start": start,
                            "end": start + current_duration
                        }
                        raw_results.append(result)
                        self.log_text(f"Analyzing {start:.1f}s - {start + current_duration:.1f}s... OK ({fb_bpm:.1f} BPM)")
                    else:
                        self.log_text(f"Analyzing {start:.1f}s - {start + current_duration:.1f}s... Failed")
                else:
                    self.log_text(f"Analyzing {start:.1f}s - {start + current_duration:.1f}s... Too few kicks")
                
                start += step
                QApplication.processEvents()

            if not raw_results:
                self.log_text("No BPM detected in the sections.")
                return

            consolidated = []
            current_group = [raw_results[0]]

            for res in raw_results[1:]:
                if abs(res["bpm"] - current_group[-1]["bpm"]) <= 1.5:
                    current_group.append(res)
                else:
                    consolidated.append(current_group)
                    current_group = [res]
            consolidated.append(current_group)

            self.log_text("")
            self.log_text("--- CONSOLIDATED RESULT ---")

            final_segments = []
            best_global = None

            for group in consolidated:
                g_start = group[0]["start"]
                g_end = group[-1]["end"]
                best_in_group = max(group, key=lambda x: x["fitness"])
                
                final_segments.append({
                    "start": g_start,
                    "end": g_end,
                    "bpm": best_in_group["bpm"],
                    "offset": best_in_group["offset"],
                    "fitness": best_in_group["fitness"]
                })
                
                self.log_text(f"{g_start:.2f}s to {g_end:.2f}s:")
                self.log_text(f"  BPM: {best_in_group['bpm']:.6f} | Offset: {best_in_group['offset'] * 1000.0:.2f}ms")
                
                if best_global is None or best_in_group["fitness"] > best_global["fitness"]:
                    best_global = best_in_group

            self.detected_segments = final_segments
            self.draw_segment_markers(final_segments)

            if best_global:
                self.bpm_input.setText(f"{best_global['bpm']:.6f}")
                self.offset_input.setText(f"{best_global['offset'] * 1000.0:.2f}")
                self.draw_beats(best_global["bpm"], best_global["offset"])
                self.log_text("")
                self.log_text(f"Best global section ({best_global['start']:.2f}s - {best_global['end']:.2f}s) applied to the grid.")
        finally:
            self.centralWidget().setEnabled(True)

    def detect_and_draw_kicks(self):
        if not self.validate_audio():
            return
        try:
            sensitivity = float(self.sensitivity_input.text().strip())
            min_gap = float(self.kick_gap_input.text().strip())
            segment_size = float(self.segment_input.text().strip())
            if min_gap <= 0 or segment_size <= 0:
                raise ValueError()
        except Exception:
            QMessageBox.critical(self, "Error", "Sensitivity, distance, and section must be valid numbers.")
            return

        self.centralWidget().setEnabled(False)
        try:
            self.log_text("")
            self.log_text("Detecting kicks...")
            QApplication.processEvents()

            if self.adaptive_check.isChecked():
                detected = []
                start = 0.0
                while start < self.duration:
                    end = min(self.duration, start + segment_size)
                    detected.extend(detect_kicks_range(self.raw_samples, self.sample_rate, start, end, min_gap=min_gap, sensitivity=sensitivity))
                    start += segment_size
                    QApplication.processEvents()

                unique_detected = []
                for time_value, strength in sorted(detected, key=lambda item: item[1], reverse=True):
                    if all(abs(time_value - existing[0]) >= min_gap for existing in unique_detected):
                        unique_detected.append((time_value, strength))
                unique_detected.sort(key=lambda item: item[0])
                detected = unique_detected
            else:
                detected = detect_kicks(self.raw_samples, self.sample_rate, min_gap=min_gap, sensitivity=sensitivity)

            if detected:
                strengths = np.array([kick[1] for kick in detected], dtype=np.float32)
                minimum_strength = np.percentile(strengths, 18)
                self.all_kicks = [kick for kick in detected if kick[1] >= minimum_strength]
                self.kicks = list(self.all_kicks)
            else:
                self.all_kicks = []
                self.kicks = []

            self.draw_kicks(self.kicks)
            self.draw_kick_strength_curve(self.kicks)
            self.prepare_metronome_playback()

            self.log_text(f"Kicks detected: {len(self.kicks)}")
            if self.kicks:
                strongest = max(self.kicks, key=lambda item: item[1])
                first = self.kicks[0]
                self.log_text(f"First kick: {first[0]:.6f}s")
                self.log_text(f"Strongest kick: {strongest[0]:.6f}s")
                self.log_text("Metronome on kicks enabled.")
            else:
                self.log_text("No kicks detected.")
        finally:
            self.centralWidget().setEnabled(True)

    def apply_primary_kick_mode(self):
        if not self.kicks:
            self.detect_and_draw_kicks()
        if not self.kicks:
            return
        try:
            bpm = float(self.bpm_input.text().strip())
        except Exception:
            bpm = 0.0
        if bpm <= 0:
            QMessageBox.warning(self, "Warning", "Detect or enter the BPM first to use primary kick mode.")
            return
        self.kicks = filter_primary_kicks(self.all_kicks or self.kicks, bpm)
        self.draw_kicks(self.kicks)
        self.draw_kick_strength_curve(self.kicks)
        self.prepare_metronome_playback()
        self.log_text("")
        self.log_text(f"Primary kick mode applied: {len(self.kicks)} kicks kept.")

    def draw_toms(self, toms):
        self.clear_tom_lines()
        self.clear_beat_lines()
        self.clear_kick_lines()
        self.set_metronome_mode("toms")
        if not toms:
            return
        strengths = np.array([tom[1] for tom in toms], dtype=np.float32)
        maximum = np.max(strengths) + 1e-9
        for time_value, strength in toms:
            normalized = strength / maximum
            width = 0.8 + normalized * 2.2
            line = pg.InfiniteLine(pos=time_value, angle=90, pen=pg.mkPen("#ffaa00", width=width))
            self.wave_plot.addItem(line)
            self.tom_lines.append(line)

    def detect_and_draw_toms(self):
        if not self.validate_audio():
            return
        try:
            sensitivity = float(self.sensitivity_input.text().strip())
            min_gap = float(self.kick_gap_input.text().strip())
            segment_size = float(self.segment_input.text().strip())
            if min_gap <= 0 or segment_size <= 0:
                raise ValueError()
        except Exception:
            QMessageBox.critical(self, "Error", "Sensitivity, distance, and section must be valid numbers.")
            return

        self.centralWidget().setEnabled(False)
        try:
            self.log_text("")
            self.log_text("Detecting toms (drums)...")
            QApplication.processEvents()

            if self.adaptive_check.isChecked():
                detected = []
                start = 0.0
                while start < self.duration:
                    end = min(self.duration, start + segment_size)
                    start_sample = max(0, int(start * self.sample_rate))
                    end_sample = min(len(self.raw_samples), int(end * self.sample_rate))
                    segment = self.raw_samples[start_sample:end_sample]
                    toms_detected = detect_toms(segment, self.sample_rate, min_gap=min_gap, sensitivity=sensitivity)
                    detected.extend([(tv + start, st) for tv, st in toms_detected])
                    start += segment_size
                    QApplication.processEvents()

                unique_detected = []
                for time_value, strength in sorted(detected, key=lambda item: item[1], reverse=True):
                    if all(abs(time_value - existing[0]) >= min_gap for existing in unique_detected):
                        unique_detected.append((time_value, strength))
                unique_detected.sort(key=lambda item: item[0])
                detected = unique_detected
            else:
                detected = detect_toms(self.raw_samples, self.sample_rate, min_gap=min_gap, sensitivity=sensitivity)

            if detected:
                strengths = np.array([tom[1] for tom in detected], dtype=np.float32)
                minimum_strength = np.percentile(strengths, 18)
                self.toms = [tom for tom in detected if tom[1] >= minimum_strength]
            else:
                self.toms = []

            self.draw_toms(self.toms)
            
            self.log_text(f"Toms detected: {len(self.toms)}")
            if self.toms:
                self.log_text("The toms were drawn on the graph (orange lines).")
            else:
                self.log_text("No toms detected.")
        finally:
            self.centralWidget().setEnabled(True)

    def auto_fit_bpm_offset(self):
        if not self.validate_audio():
            return
        try:
            initial_bpm = float(self.bpm_input.text().strip())
        except Exception:
            initial_bpm = 0.0

        if not self.kicks:
            self.detect_and_draw_kicks()

        if not self.kicks:
            QMessageBox.warning(self, "Warning", "No kicks were detected.")
            return

        fitted_bpm, fitted_offset, score = fit_bpm_offset_to_kicks(self.kicks, initial_bpm)
        self.bpm_input.setText(f"{fitted_bpm:.6f}")
        self.offset_input.setText(f"{fitted_offset * 1000.0:.2f}")
        self.draw_beats(fitted_bpm, fitted_offset)

        self.log_text("")
        self.log_text("BPM/offset adjusted using kicks:")
        self.log_text(f"Previous BPM: {initial_bpm:.6f}")
        self.log_text(f"Adjusted BPM: {fitted_bpm:.6f}")
        self.log_text(f"Adjusted offset: {fitted_offset * 1000.0:.2f}ms")
        self.log_text(f"Fit score: {score:.6f}")

    def sync_offset_with_kick(self):
        self.auto_fit_bpm_offset()

    def set_metronome_mode(self, mode):
        self.metronome_mode = mode
        if mode == "kicks":
            self.kick_metronome_button.setText("Metronome: kicks")
        elif mode == "toms":
            self.kick_metronome_button.setText("Metronome: toms")
        elif mode == "bpm":
            self.kick_metronome_button.setText("Metronome: BPM")
        self.prepare_metronome_playback()

    def toggle_metronome_mode(self):
        if self.metronome_mode == "kicks":
            self.set_metronome_mode("toms")
            self.log_text("Metronome configured to play detected toms.")
        elif self.metronome_mode == "toms":
            self.set_metronome_mode("bpm")
            self.log_text("Metronome configured to play the BPM grid.")
        else:
            self.set_metronome_mode("kicks")
            self.log_text("Metronome configured to play detected kicks.")

    def update_click_volume(self):
        volume = self.click_volume_slider.value() / 200.0
        self.kick_sound.setVolume(volume)
        self.tom_sound.setVolume(volume)

    def adjust_offset(self, amount):
        try:
            offset = float(self.offset_input.text().strip()) / 1000.0
        except Exception:
            offset = 0.0
        offset += amount
        self.offset_input.setText(f"{offset * 1000.0:.2f}")
        self.apply_beats()

    def adjust_sensitivity(self, amount):
        try:
            value = float(self.sensitivity_input.text().strip())
        except Exception:
            value = 1.75
        value = max(0.20, value + amount)
        self.sensitivity_input.setText(f"{value:.2f}")
        if self.audio_path:
            self.detect_and_draw_kicks()

    def adjust_gap(self, amount):
        try:
            value = float(self.kick_gap_input.text().strip())
        except Exception:
            value = 0.28
        value = max(0.05, value + amount)
        self.kick_gap_input.setText(f"{value:.2f}")
        if self.audio_path:
            self.detect_and_draw_kicks()

    def clear_beat_lines(self):
        for line in self.beat_lines:
            self.wave_plot.removeItem(line)
        self.beat_lines = []

    def clear_kick_lines(self):
        for line in self.kick_lines:
            self.wave_plot.removeItem(line)
        self.kick_lines = []

    def clear_tom_lines(self):
        for line in self.tom_lines:
            self.wave_plot.removeItem(line)
        self.tom_lines = []

    def clear_segment_lines(self):
        for line in self.segment_lines:
            self.wave_plot.removeItem(line)
        self.segment_lines = []

    def draw_beats(self, bpm, offset):
        self.clear_beat_lines()
        self.clear_kick_lines()
        self.clear_tom_lines()
        self.metronome_bpm = bpm
        self.metronome_offset = offset
        self.set_metronome_mode("bpm")
        if bpm <= 0:
            return
        interval = 60.0 / bpm
        beat = offset % interval
        if beat < 0:
            beat += interval
        
        shift = int(round((offset - beat) / interval))
        index = -shift

        while beat <= self.duration:
            if index % 4 == 0:
                pen = pg.mkPen("#ffffff", width=1.6)
            else:
                pen = pg.mkPen("#aaaaaa", width=0.8)
            line = pg.InfiniteLine(pos=beat, angle=90, pen=pen)
            self.wave_plot.addItem(line)
            self.beat_lines.append(line)
            beat += interval
            index += 1

    def draw_kicks(self, kicks):
        self.clear_kick_lines()
        self.clear_beat_lines()
        self.clear_tom_lines()
        self.set_metronome_mode("kicks")
        if not kicks:
            return
        strengths = np.array([kick[1] for kick in kicks], dtype=np.float32)
        maximum = np.max(strengths) + 1e-9
        for time_value, strength in kicks:
            normalized = strength / maximum
            width = 0.8 + normalized * 2.2
            line = pg.InfiniteLine(pos=time_value, angle=90, pen=pg.mkPen("#00ff66", width=width))
            self.wave_plot.addItem(line)
            self.kick_lines.append(line)

    def draw_kick_strength_curve(self, kicks):
        if not kicks:
            self.kick_curve.setData([], [])
            return
        times = np.array([kick[0] for kick in kicks], dtype=np.float32)
        strengths = np.array([kick[1] for kick in kicks], dtype=np.float32)
        if np.max(strengths) > 0:
            strengths = strengths / np.max(strengths)
        self.kick_curve.setData(times, strengths, pen=None, symbol="o", symbolSize=5, symbolBrush="#00ff66")

    def draw_segment_markers(self, results):
        self.clear_segment_lines()
        for result in results:
            line = pg.InfiniteLine(pos=result["start"], angle=90, pen=pg.mkPen("#00d0ff", width=2))
            self.wave_plot.addItem(line)
            self.segment_lines.append(line)

    def apply_beats(self):
        try:
            bpm = float(self.bpm_input.text().strip())
            offset = float(self.offset_input.text().strip()) / 1000.0
            if bpm <= 0:
                raise ValueError()
        except Exception:
            QMessageBox.critical(self, "Error", "Enter a valid BPM and offset.")
            return
        self.draw_beats(bpm, offset)

    def seek_to_plot_click(self, event):
        if not self.validate_audio():
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.wave_plot.sceneBoundingRect().contains(event.scenePos()):
            return
        mouse_point = self.wave_plot.plotItem.vb.mapSceneToView(event.scenePos())
        position = mouse_point.x()
        if position < 0:
            position = 0
        if position > self.duration:
            position = self.duration
        self.player.setPosition(int(position * 1000))
        self.cursor_line.setPos(position)
        self.update_position_label(position)
        self.prepare_metronome_playback()

    def prepare_metronome_playback(self):
        position = self.player.position() / 1000.0
        self.prepare_kick_metronome(position)
        self.prepare_tom_metronome(position)
        self.prepare_bpm_metronome(position)

    def prepare_kick_metronome(self, position):
        self.next_kick_index = 0
        if not self.kicks:
            return
        while self.next_kick_index < len(self.kicks) and self.kicks[self.next_kick_index][0] < position:
            self.next_kick_index += 1

    def prepare_tom_metronome(self, position):
        self.next_tom_index = 0
        if not self.toms:
            return
        while self.next_tom_index < len(self.toms) and self.toms[self.next_tom_index][0] < position:
            self.next_tom_index += 1

    def prepare_bpm_metronome(self, position):
        if self.metronome_bpm <= 0:
            self.next_beat_time = None
            return
        interval = 60.0 / self.metronome_bpm
        beat = self.metronome_offset % interval
        if beat < 0:
            beat += interval
        while beat < position:
            beat += interval
        self.next_beat_time = beat

    def play_metronome_if_needed(self, position):
        if self.metronome_mode == "bpm":
            if self.next_beat_time is None or abs(position - self.next_beat_time) > (120.0 / self.metronome_bpm if self.metronome_bpm > 0 else 1.0):
                self.prepare_bpm_metronome(position)
            self.play_bpm_metronome_if_needed(position)
        elif self.metronome_mode == "kicks" and self.kicks:
            self.play_kick_metronome_if_needed(position)
        elif self.metronome_mode == "toms" and self.toms:
            self.play_tom_metronome_if_needed(position)

    def play_kick_metronome_if_needed(self, position):
        if not self.kicks:
            return
        while self.next_kick_index < len(self.kicks) and position >= self.kicks[self.next_kick_index][0]:
            self.kick_sound.stop()
            self.kick_sound.play()
            self.metronome_boost = 1.2
            self.next_kick_index += 1

    def play_tom_metronome_if_needed(self, position):
        if not self.toms:
            return
        while self.next_tom_index < len(self.toms) and position >= self.toms[self.next_tom_index][0]:
            self.tom_sound.stop()
            self.tom_sound.play()
            self.metronome_boost = 1.2
            self.next_tom_index += 1

    def play_bpm_metronome_if_needed(self, position):
        if self.metronome_bpm <= 0 or self.next_beat_time is None:
            return
        interval = 60.0 / self.metronome_bpm
        while position >= self.next_beat_time:
            beat_idx = int(round((self.next_beat_time - self.metronome_offset) / interval))
            if beat_idx % 4 == 0:
                self.kick_sound.stop()
                self.kick_sound.play()
            else:
                self.tom_sound.stop()
                self.tom_sound.play()
            self.metronome_boost = 1.2
            self.next_beat_time += interval

    def toggle_play(self):
        if not self.validate_audio():
            return
        self.update_click_volume()
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_button.setText("Play")
            self.timer.stop()
        else:
            self.prepare_metronome_playback()
            self.player.play()
            self.play_button.setText("Pause")
            self.timer.start()

    def stop_audio(self):
        self.player.stop()
        self.play_button.setText("Play")
        self.timer.stop()
        self.next_beat_time = None
        self.next_kick_index = 0
        self.next_tom_index = 0
        self.cursor_line.setPos(0)
        self.update_position_label(0)

    def update_spectrum(self, position):
        if len(self.raw_samples) == 0 or self.sample_rate <= 0:
            return
        idx = int(position * self.sample_rate)
        frame_size = self.spectrum_points
        if idx < 0 or idx + frame_size > len(self.raw_samples):
            self.spectrum_curve.setData(np.zeros(self.spectrum_points))
            return
        frame = self.raw_samples[idx:idx + frame_size]
        window = np.hanning(frame_size)
        scale = 0.25 + (self.metronome_boost * 2.2)
        if self.metronome_boost > 0.001:
            self.metronome_boost *= 0.75
        else:
            self.metronome_boost = 0.0
        base_y = np.clip(frame * window * scale, -1.1, 1.1)
        self.spectrum_curve.setData(base_y)

    def update_cursor(self):
        position = self.player.position() / 1000.0
        self.cursor_line.setPos(position)
        self.update_position_label(position)
        self.play_metronome_if_needed(position)
        self.update_spectrum(position)
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("Play")
            self.timer.stop()

    def update_position_label(self, position):
        self.position_label.setText(f"{self.format_time(position)} / {self.format_time(self.duration)}")

    def format_time(self, seconds):
        seconds = max(0, int(seconds))
        minutes = seconds // 60
        rest = seconds % 60
        return f"{minutes:02d}:{rest:02d}"

    def apply_zoom(self):
        value = self.zoom_combo.currentText()
        if not self.duration:
            return
        if value == "Full":
            self.wave_plot.setXRange(0, self.duration, padding=0)
            return
        seconds = float(value.replace("s", ""))
        position = self.player.position() / 1000.0
        start = max(0, position - seconds / 2)
        end = min(self.duration, start + seconds)
        if end - start < seconds:
            start = max(0, end - seconds)
        self.wave_plot.setXRange(start, end, padding=0)

    def export_csv(self):
        if not self.validate_audio():
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            os.path.join(BASE_DIR, "resultado_bpm_kicks.csv"),
            "CSV (*.csv)"
        )
        if not path:
            return
        bpm = self.bpm_input.text().strip()
        offset = self.offset_input.text().strip()
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["audio", self.audio_path])
            writer.writerow(["bpm", bpm])
            writer.writerow(["offset", offset])
            writer.writerow([])
            writer.writerow(["kick_time", "strength"])
            for time_value, strength in self.kicks:
                writer.writerow([f"{time_value:.6f}", f"{strength:.6f}"])
        self.log_text(f"CSV exported: {path}")

    def export_json(self):
        if not self.validate_audio():
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export JSON",
            os.path.join(BASE_DIR, "resultado_bpm_kicks.json"),
            "JSON (*.json)"
        )
        if not path:
            return
        data = {
            "audio": self.audio_path,
            "duration": self.duration,
            "bpm": self.bpm_input.text().strip(),
            "offset": self.offset_input.text().strip(),
            "kicks": [
                {
                    "time": time_value,
                    "strength": strength
                }
                for time_value, strength in self.kicks
            ]
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        self.log_text(f"JSON exported: {path}")

    def export_pagoda(self):
        if not self.validate_audio():
            return
        
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            QMessageBox.critical(self, "Error", "FFmpeg was not found.")
            return

        try:
            bpm_text = self.bpm_input.text().strip().replace(',', '.')
            offset_text = self.offset_input.text().strip().replace(',', '.')
            bpm = float(bpm_text)
            offset_ms = float(offset_text)
        except Exception:
            QMessageBox.critical(self, "Error", "BPM and Offset must be valid numbers.")
            return
            
        base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
        if " - " in base_name:
            parts = base_name.split(" - ", 1)
            artist = parts[0].strip()
            song = parts[1].strip()
        else:
            artist = "Unknown"
            song = base_name.strip()
            
        folder_name = f"{artist} - {song}"
        pagoda_path = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Pagoda", "Saved", "ImportedSongs", folder_name)
        os.makedirs(pagoda_path, exist_ok=True)
        
        ogg_path_sys = os.path.join(pagoda_path, "Audio.ogg")
        ogg_path_json = ogg_path_sys.replace("\\", "/")
        
        self.log_text(f"Converting to OGG in folder: {pagoda_path}...")
        QApplication.processEvents()
        
        process = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", self.audio_path, "-map", "0:a:0", "-map_metadata", "-1", "-c:a", "libvorbis", "-ac", "2", "-ar", "44100", "-q:a", "5", ogg_path_sys], text=True, capture_output=True, env=build_env())
        if process.returncode != 0:
            error_msg = process.stderr.strip() if process.stderr else "Unknown error during conversion."
            QMessageBox.critical(self, "Conversion Error", error_msg)
            return

        md5 = hashlib.md5()
        with open(ogg_path_sys, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
                
        meta_data = {
            "version": 1,
            "uniqueId": random.randint(1, 4294967295),
            "songName": song,
            "performedBy": [artist],
            "writtenBy": [],
            "seed": random.randint(1, 4294967295),
            "tempo": bpm,
            "beatOffset": int(round(offset_ms)),
            "startSongOffset": 0.0,
            "endSongOffset": 0.0,
            "uEAssetName": folder_name,
            "originalAudioFileHash": md5.hexdigest(),
            "originalAudioFilePath": ogg_path_json
        }
        
        custom_sections = []
        if len(self.detected_segments) > 1:
            for seg in self.detected_segments:
                if seg["start"] == 0.0 and abs(seg["bpm"] - bpm) < 0.1:
                    continue
                custom_sections.append({
                    "tempo": seg["bpm"],
                    "startAbsoluteTime": seg["start"]
                })
        
        if custom_sections:
            meta_data["customTempoSections"] = custom_sections
            
        json_path = os.path.join(pagoda_path, "Meta.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
            
        self.log_text(f"Pagoda export completed: {json_path}")
        QMessageBox.information(self, "Success", f"Song successfully exported to Pagoda!\n\n{pagoda_path}")

    def export_midi(self):
        if not self.validate_audio():
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export MIDI",
            os.path.join(BASE_DIR, f"metronomo_{self.metronome_mode}.mid"),
            "MIDI file (*.mid)"
        )
        if not path:
            return

        try:
            bpm = float(self.bpm_input.text().strip().replace(',', '.'))
        except Exception:
            bpm = 120.0

        try:
            offset_ms = float(self.offset_input.text().strip().replace(',', '.'))
            offset = offset_ms / 1000.0
        except Exception:
            offset = 0.0

        events = []
        note_pitch = 36

        if self.metronome_mode == "kicks":
            if not self.kicks:
                QMessageBox.warning(self, "Warning", "No kicks detected to export.")
                return
            max_s = max([s for _, s in self.kicks]) + 1e-9 if self.kicks else 1.0
            events = [(t, s / max_s) for t, s in self.kicks]
            note_pitch = 36

        elif self.metronome_mode == "toms":
            if not self.toms:
                QMessageBox.warning(self, "Warning", "No toms detected to export.")
                return
            max_s = max([s for _, s in self.toms]) + 1e-9 if self.toms else 1.0
            events = [(t, s / max_s) for t, s in self.toms]
            note_pitch = 45

        else:
            if bpm <= 0:
                QMessageBox.warning(self, "Warning", "Enter a valid BPM to export.")
                return
            interval = 60.0 / bpm
            beat = offset % interval
            if beat < 0:
                beat += interval
            shift = int(round((offset - beat) / interval))
            idx = -shift
            
            events = []
            while beat <= self.duration:
                vel = 1.0 if idx % 4 == 0 else 0.7
                pitch = 36 if idx % 4 == 0 else 45
                events.append((beat, vel, pitch))
                beat += interval
                idx += 1

        create_midi_file(path, bpm, events, default_pitch=note_pitch)
        self.log_text(f"MIDI exported successfully: {path}")
        QMessageBox.information(self, "Success", f"MIDI file successfully generated at:\n{path}")

    def closeEvent(self, event):
        self.stop_audio()
        if self.temp_wav_path and os.path.isfile(self.temp_wav_path):
            try:
                os.remove(self.temp_wav_path)
            except Exception:
                pass
        for path in [self.kick_path, self.tom_path]:
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

