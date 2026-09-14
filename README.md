<div align="center">

# 🎛️ BPM Auto Finder

### Automatic BPM, offset, and kick detection for music analysis

</div>

---

**BPM Auto Finder** is a Python tool designed to analyze music and automatically identify the **BPM**, **offset**, and **kick drum positions** throughout a track.

It also provides tools to:

* Visualize the audio waveform
* Analyze BPM changes across different sections
* Manually adjust detected results
* Verify synchronization using a built-in metronome

### Export Formats

Analysis results can be exported in:

* **CSV**
* **JSON**
* **MIDI**

---  

### Dead as Disco support
Originally, this project was made for the game Dead as Disco, which is a rhythm fighting game where you can select any song downloaded on your computer. The problem was that, at the time, Dead as Disco didn't have an automatic BPM detector for downloaded songs, so this project was created. Now the game has this feature, but, at least at the time of writing, this program is still more accurate at detecting BPM than Dead as Disco's built-in BPM detector.

To export your songs to Dead as Disco just click the "Export Pagoda" option at the top right corner of the program.

---

### Dependencies
* FFmpeg
