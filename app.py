import streamlit as st
import librosa
import numpy as np
import base64

# Function for base64 encoding a local image
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Use your local image file name here (must have the extension, e.g., .jpg)
image_file_name = "piano_blur.jpg"
base64_img = get_base64_of_image(image_file_name)

page_bg_img = f'''
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{base64_img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    /* Optional: filter: blur(3px); */
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Overlay for readable content
overlay_css = """
<style>
.block-container {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 15px;
    padding: 36px 36px 24px 36px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.13);
}
h1, h2, h3, label {
    color: #222 !important;
}
</style>
"""
st.markdown(overlay_css, unsafe_allow_html=True)

st.title("🎵 Piano/Instrument Note Detector")
st.markdown("Upload an audio file (.wav) to detect played piano notes and chords below.")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

def freq_to_note(freq):
    if freq == 0:
        return "-"
    A4 = 440.0
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    semitones = int(np.round(12 * np.log2(freq / A4)))
    note_index = (semitones + 9) % 12  
    octave = 4 + ((semitones + 9) // 12)
    return f"{notes[note_index]}{octave}"

if uploaded_file:
    y, sr = librosa.load(uploaded_file)
    st.audio(uploaded_file)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    notes = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            note = freq_to_note(pitch)
            notes.append(note)
    st.markdown("### Detected Notes (sample):")
    st.write(notes[:20])

    # --- Chord Detection ---

    # Chord intervals (in semitones)
    basic_chords = {
        "Major": {0, 4, 7},
        "Minor": {0, 3, 7},
        "Diminished": {0, 3, 6},
        "Augmented": {0, 4, 8},
    }

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def note_to_midi(note):
        if note == '-':
            return None
        name = note[:-1]
        octave = int(note[-1])
        base = note_names.index(name)
        return (octave + 1) * 12 + base

    def detect_chord(notes_in_frame):
        midi_notes = sorted(n for n in (note_to_midi(n) for n in notes_in_frame) if n is not None)
        if not midi_notes:
            return "No chord"
        for root_note in midi_notes:
            intervals = {(n - root_note) % 12 for n in midi_notes}
            for chord_name, pattern in basic_chords.items():
                if pattern.issubset(intervals):
                    root_name_str = note_names[(root_note - 12) % 12]
                    return f"{root_name_str} {chord_name}"
        return "Unknown"

    chord_list = []
    for i in range(pitches.shape[1]):
        frame_notes = []
        for j in range(pitches.shape[0]):
            if magnitudes[j, i] > 0.1:
                pitch = pitches[j, i]
                if pitch > 0:
                    frame_notes.append(freq_to_note(pitch))
        if frame_notes:
            chord = detect_chord(frame_notes)
            chord_list.append(chord)

    st.markdown("### Detected Chords (sample):")
    st.write(chord_list[:20])
