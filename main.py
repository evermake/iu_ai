"""
Program to generate accompaniment for monophonic melodies using genetic algorithm.

Author: Vladislav Deryabkin
Date: November 2022
"""

import argparse
import logging
import re
from collections import defaultdict, Counter
from enum import Enum
from math import ceil
from typing import NamedTuple, Callable, List, Dict, Set, Optional, Tuple

import mido
import numpy as np


logger = logging.getLogger(__name__)


class Note:
    """
    Note is a class that represents a musical note with utility methods.

    It can be created by number (1 - C, 2 - C#, etc.),
    by name (C, C#, etc.) or by MIDI number.

    Note can have an octave that allows to convert it MIDI number,
    but also it can be a note without octave, which is used for
    representing chords.

    Note can be added to an integer to get a next note.
    """

    NUM_BY_PITCH = {
        "C": 1,
        "D": 3,
        "E": 5,
        "F": 6,
        "G": 8,
        "A": 10,
        "B": 12,
    }
    NAME_BY_NUM = {
        1: "C",
        2: "C#",
        3: "D",
        4: "D#",
        5: "E",
        6: "F",
        7: "F#",
        8: "G",
        9: "G#",
        10: "A",
        11: "A#",
        12: "B",
    }
    NAME_REGEX = re.compile(r"^([A-G])([#b]?)(\d)?$")

    def __init__(self, name: str):
        # parse note
        name = name[:1].upper() + name[1:]
        match = self.NAME_REGEX.match(name)
        if match is None:
            raise ValueError(f"invalid note name '{name}'")

        pitch, accidental, octave = match.groups()

        # adjust accidental
        num = self.NUM_BY_PITCH[pitch]
        if accidental == "#":
            num += 1
        elif accidental == "b":
            num -= 1
        num = (num - 1) % 12 + 1
        octave = int(octave) if octave is not None else None

        self._raw_name = name
        self._num = num
        self._octave = octave

    @property
    def octave(self) -> Optional[int]:
        return self._octave

    @property
    def midi(self) -> int:
        """
        Get MIDI number of the note.

        Warning: note without octave cannot be converted to MIDI number.
        """
        if self._octave is None:
            raise ValueError("note w/o octave has no midi")
        return (self._octave + 1) * 12 + self._num - 1

    @classmethod
    def from_midi(cls, midi: int) -> "Note":
        """Create note from MIDI number."""
        # clamp to 21-108
        if midi < 21:
            midi = 21
        elif midi > 108:
            midi = 108

        octave = (midi - 12) // 12
        num = midi % 12 + 1
        name = cls.NAME_BY_NUM[num]
        return cls(f"{name}{octave}")

    @classmethod
    def from_num(cls, num: int, octave: Optional[int] = None) -> "Note":
        """
        Create note from number.

        :param num: Number of the note in range [1,12], e.g. 1 - C, 2 - C#, etc.
        :param octave: Octave of the note.
        """
        num = (num - 1) % 12 + 1
        if num not in cls.NAME_BY_NUM:
            raise ValueError(f"invalid note num {num}")
        name = cls.NAME_BY_NUM[num]
        octave_str = str(octave) if octave is not None else ""
        return cls(f"{name}{octave_str}")

    @property
    def raw_name(self):
        """Note name as it was passed on initialization."""
        return self._raw_name

    @property
    def name(self) -> str:
        """Note name with octave, if its present."""
        name = self.NAME_BY_NUM[self._num]
        if self._octave is None:
            return name
        return f"{name}{self._octave}"

    @property
    def num(self) -> int:
        return self._num

    @property
    def without_octave(self) -> "Note":
        """Return a copy of this note without octave."""
        copy = Note(self.raw_name)
        copy._octave = None
        return copy

    def __repr__(self):
        return f"Note.{self.__str__()}"

    def __str__(self):
        return self.name

    def __add__(self, other: int):
        """
        Return a note that is `other` semitones higher than this note,
        if `self` has an octave, or a next note in the scale otherwise.
        """
        if not isinstance(other, int):
            raise TypeError(f"cannot add 'Note' and '{type(other).__name__}'")
        if self._octave is None:
            return self.from_num(self._num + other)
        return self.from_midi(self.midi + other)

    def __eq__(self, other):
        if not isinstance(other, Note):
            return NotImplemented
        return self._num == other._num and self._octave == other._octave

    def __hash__(self):
        return hash((self._num, self._octave))


class ChordType(Enum):
    """All possible chord types."""

    REST = 0
    MAJOR1 = 1
    MAJOR1_INV1 = 2
    MAJOR1_INV2 = 3
    MAJOR2 = 4
    MAJOR2_INV1 = 5
    MAJOR2_INV2 = 6
    MAJOR3 = 7
    MAJOR3_INV1 = 8
    MAJOR3_INV2 = 9
    MINOR1 = 10
    MINOR1_INV1 = 11
    MINOR1_INV2 = 12
    MINOR2 = 13
    MINOR2_INV1 = 14
    MINOR2_INV2 = 15
    MINOR3 = 16
    MINOR3_INV1 = 17
    MINOR3_INV2 = 18
    DIM = 19
    SUS2_1 = 20
    SUS2_2 = 21
    SUS2_3 = 22
    SUS2_4 = 23
    SUS2_5 = 24
    SUS4_1 = 25
    SUS4_2 = 26
    SUS4_3 = 27
    SUS4_4 = 28
    SUS4_5 = 29


class Key:
    """
    Base class for all keys.

    Keys are used to generate chords and analyze key of a melody.
    """

    type: str = None

    def __init__(self, tonic: Note):
        self._tonic = tonic.without_octave

    @property
    def tonic(self) -> Note:
        return self._tonic

    @property
    def short_name(self) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def scale(self) -> List[Note]:
        raise NotImplementedError

    @property
    def scale_set(self) -> Set[Note]:
        return set(self.scale)

    @classmethod
    def get_all_possible_keys(cls) -> List["Key"]:
        raise NotImplementedError

    def rate_note_stability(self, note: Note) -> float:
        """
        Return a number in range [0,1] that
        represents how stable is the note in this key.
        """
        note = note.without_octave
        tonic, supertonic, mediant, subdominant, dominant, submediant, leading = self.scale
        if note == tonic:
            return 1.0
        elif note == supertonic:
            return 0.6
        elif note == mediant:
            return 0.7
        elif note == subdominant:
            return 0.79
        elif note == dominant:
            return 0.8
        elif note == submediant:
            return 0.6
        elif note == leading:
            return 0.6
        return 0.0

    def rate_note_as_starting(self, note: Note) -> float:
        """
        Return a number (at least 1.0) that represents
        how likely is the note to be a starting note of melody in this key.
        """
        note = note.without_octave
        tonic, supertonic, mediant, subdominant, dominant, submediant, leading = self.scale
        if note == tonic:
            return 1.08
        elif note == mediant or note == subdominant:
            return 1.05
        elif note == dominant:
            return 1.1
        return 1.0

    def rate_note_as_ending(self, note: Note) -> float:
        """
        Return a number (at least 1.0) that represents
        how likely is the note to be a ending note of melody in this key.
        """
        note = note.without_octave
        tonic, supertonic, mediant, subdominant, dominant, submediant, leading = self.scale
        if note == tonic:
            return 1.2
        elif note == mediant:
            return 1.1
        elif note == dominant:
            return 1.15
        return 1.0

    def get_chord_notes(self, chord_type: ChordType, octave: int) -> List[Note]:
        raise NotImplementedError

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return other.type == self.type and other.tonic == self.tonic

    def __hash__(self):
        return hash((self.type, self.tonic))


class MajorKey(Key):
    """Implementation of major key."""

    type = "major"

    def __init__(self, tonic: Note):
        tonic = tonic.without_octave
        super().__init__(tonic)
        # define scale
        self._scale = [
            self.tonic,
            self.tonic + 2,
            self.tonic + 4,
            self.tonic + 5,
            self.tonic + 7,
            self.tonic + 9,
            self.tonic + 11,
        ]
        self._scale_set = set(self._scale)

    @property
    def short_name(self) -> str:
        return self.tonic.name

    @property
    def name(self) -> str:
        return f"{self.tonic.name} major"

    @property
    def scale(self) -> List[Note]:
        return self._scale

    @property
    def scale_set(self) -> Set[Note]:
        return self._scale_set

    @classmethod
    def get_all_possible_keys(cls) -> List["MajorKey"]:
        return [cls(Note.from_num(num)) for num in range(1, 13)]

    def get_chord_notes(self, chord_type: ChordType, octave: int) -> List[Note]:
        """
        Return a list of notes that are in the
        specified chord and octave for this key.
        """
        n1, n2, n3, n4, n5, n6, n7 = self.scale

        if chord_type == ChordType.DIM:
            root = Note.from_num(n7.num, octave)
            return [root, root + 3, root + 6]

        if chord_type in (ChordType.SUS2_1, ChordType.SUS2_2, ChordType.SUS2_3, ChordType.SUS2_4, ChordType.SUS2_5):
            if chord_type == ChordType.SUS2_1:
                root = Note.from_num(n1.num, octave)
            elif chord_type == ChordType.SUS2_2:
                root = Note.from_num(n2.num, octave)
            elif chord_type == ChordType.SUS2_3:
                root = Note.from_num(n4.num, octave)
            elif chord_type == ChordType.SUS2_4:
                root = Note.from_num(n5.num, octave)
            elif chord_type == ChordType.SUS2_5:
                root = Note.from_num(n6.num, octave)
            else:
                raise ValueError("invalid chord type")
            return [root, root + 2, root + 7]

        if chord_type in (ChordType.SUS4_1, ChordType.SUS4_2, ChordType.SUS4_3, ChordType.SUS4_4, ChordType.SUS4_5):
            if chord_type == ChordType.SUS4_1:
                root = Note.from_num(n1.num, octave)
            elif chord_type == ChordType.SUS4_2:
                root = Note.from_num(n2.num, octave)
            elif chord_type == ChordType.SUS4_3:
                root = Note.from_num(n3.num, octave)
            elif chord_type == ChordType.SUS4_4:
                root = Note.from_num(n5.num, octave)
            elif chord_type == ChordType.SUS4_5:
                root = Note.from_num(n6.num, octave)
            else:
                raise ValueError("invalid chord type")
            return [root, root + 5, root + 7]

        if chord_type in (ChordType.MAJOR1, ChordType.MAJOR1_INV1, ChordType.MAJOR1_INV2):
            root = Note.from_num(n1.num, octave)
        elif chord_type in (ChordType.MAJOR2, ChordType.MAJOR2_INV1, ChordType.MAJOR2_INV2):
            root = Note.from_num(n4.num, octave)
        elif chord_type in (ChordType.MAJOR3, ChordType.MAJOR3_INV1, ChordType.MAJOR3_INV2):
            root = Note.from_num(n5.num, octave)
        elif chord_type in (ChordType.MINOR1, ChordType.MINOR1_INV1, ChordType.MINOR1_INV2):
            root = Note.from_num(n2.num, octave)
        elif chord_type in (ChordType.MINOR2, ChordType.MINOR2_INV1, ChordType.MINOR2_INV2):
            root = Note.from_num(n3.num, octave)
        elif chord_type in (ChordType.MINOR3, ChordType.MINOR3_INV1, ChordType.MINOR3_INV2):
            root = Note.from_num(n6.num, octave)
        else:
            raise ValueError("invalid chord type")

        if chord_type in (ChordType.MAJOR1, ChordType.MAJOR2, ChordType.MAJOR3):
            return [root, root + 4, root + 7]
        elif chord_type in (ChordType.MAJOR1_INV1, ChordType.MAJOR2_INV1, ChordType.MAJOR3_INV1):
            return [root + 12, root + 4, root + 7]
        elif chord_type in (ChordType.MAJOR1_INV2, ChordType.MAJOR2_INV2, ChordType.MAJOR3_INV2):
            return [root + 12, root + 16, root + 7]
        elif chord_type in (ChordType.MINOR1, ChordType.MINOR2, ChordType.MINOR3):
            return [root, root + 3, root + 7]
        elif chord_type in (ChordType.MINOR1_INV1, ChordType.MINOR2_INV1, ChordType.MINOR3_INV1):
            return [root + 12, root + 3, root + 7]
        elif chord_type in (ChordType.MINOR1_INV2, ChordType.MINOR2_INV2, ChordType.MINOR3_INV2):
            return [root + 12, root + 15, root + 7]

    def __repr__(self):
        return f"MajorKey({self.tonic!r})"


class MinorKey(Key):
    """Implementation of a minor key."""

    type = "minor"

    def __init__(self, tonic: Note):
        tonic = tonic.without_octave
        super().__init__(tonic)
        # define the scale
        self._scale = [
            self.tonic,
            self.tonic + 2,
            self.tonic + 3,
            self.tonic + 5,
            self.tonic + 7,
            self.tonic + 8,
            self.tonic + 10,
        ]
        self._scale_set = set(self._scale)

    @property
    def short_name(self) -> str:
        return f"{self.tonic.name}m"

    @property
    def name(self) -> str:
        return f"{self.tonic.name} minor"

    @property
    def scale(self) -> List[Note]:
        return self._scale

    @property
    def scale_set(self) -> Set[Note]:
        return self._scale_set

    @classmethod
    def get_all_possible_keys(cls) -> List["MinorKey"]:
        return [cls(Note.from_num(num)) for num in range(1, 13)]

    def get_chord_notes(self, chord_type: ChordType, octave: int) -> List[Note]:
        """
        Return a list of notes that are in the
        specified chord and octave for this key.
        """
        n1, n2, n3, n4, n5, n6, n7 = self.scale

        if chord_type == ChordType.DIM:
            root = Note.from_num(n2.num, octave)
            return [root, root + 3, root + 6]

        if chord_type in (ChordType.SUS2_1, ChordType.SUS2_2, ChordType.SUS2_3, ChordType.SUS2_4, ChordType.SUS2_5):
            if chord_type == ChordType.SUS2_1:
                root = Note.from_num(n1.num, octave)
            elif chord_type == ChordType.SUS2_2:
                root = Note.from_num(n3.num, octave)
            elif chord_type == ChordType.SUS2_3:
                root = Note.from_num(n4.num, octave)
            elif chord_type == ChordType.SUS2_4:
                root = Note.from_num(n6.num, octave)
            elif chord_type == ChordType.SUS2_5:
                root = Note.from_num(n7.num, octave)
            else:
                raise ValueError("invalid chord type")
            return [root, root + 2, root + 7]

        if chord_type in (ChordType.SUS4_1, ChordType.SUS4_2, ChordType.SUS4_3, ChordType.SUS4_4, ChordType.SUS4_5):
            if chord_type == ChordType.SUS4_1:
                root = Note.from_num(n1.num, octave)
            elif chord_type == ChordType.SUS4_2:
                root = Note.from_num(n3.num, octave)
            elif chord_type == ChordType.SUS4_3:
                root = Note.from_num(n4.num, octave)
            elif chord_type == ChordType.SUS4_4:
                root = Note.from_num(n5.num, octave)
            elif chord_type == ChordType.SUS4_5:
                root = Note.from_num(n7.num, octave)
            else:
                raise ValueError("invalid chord type")
            return [root, root + 5, root + 7]

        if chord_type in (ChordType.MAJOR1, ChordType.MAJOR1_INV1, ChordType.MAJOR1_INV2):
            root = Note.from_num(n3.num, octave)
        elif chord_type in (ChordType.MAJOR2, ChordType.MAJOR2_INV1, ChordType.MAJOR2_INV2):
            root = Note.from_num(n6.num, octave)
        elif chord_type in (ChordType.MAJOR3, ChordType.MAJOR3_INV1, ChordType.MAJOR3_INV2):
            root = Note.from_num(n7.num, octave)
        elif chord_type in (ChordType.MINOR1, ChordType.MINOR1_INV1, ChordType.MINOR1_INV2):
            root = Note.from_num(n1.num, octave)
        elif chord_type in (ChordType.MINOR2, ChordType.MINOR2_INV1, ChordType.MINOR2_INV2):
            root = Note.from_num(n4.num, octave)
        elif chord_type in (ChordType.MINOR3, ChordType.MINOR3_INV1, ChordType.MINOR3_INV2):
            root = Note.from_num(n5.num, octave)
        else:
            raise ValueError("invalid chord type")

        if chord_type in (ChordType.MAJOR1, ChordType.MAJOR2, ChordType.MAJOR3):
            return [root, root + 4, root + 7]
        elif chord_type in (ChordType.MAJOR1_INV1, ChordType.MAJOR2_INV1, ChordType.MAJOR3_INV1):
            return [root + 12, root + 4, root + 7]
        elif chord_type in (ChordType.MAJOR1_INV2, ChordType.MAJOR2_INV2, ChordType.MAJOR3_INV2):
            return [root + 12, root + 16, root + 7]
        elif chord_type in (ChordType.MINOR1, ChordType.MINOR2, ChordType.MINOR3):
            return [root, root + 3, root + 7]
        elif chord_type in (ChordType.MINOR1_INV1, ChordType.MINOR2_INV1, ChordType.MINOR3_INV1):
            return [root + 12, root + 3, root + 7]
        elif chord_type in (ChordType.MINOR1_INV2, ChordType.MINOR2_INV2, ChordType.MINOR3_INV2):
            return [root + 12, root + 15, root + 7]

    def __repr__(self):
        return f"MinorKey({self.tonic!r})"


######################

# Time units:
# bar = 4 beats
# click = 1/4 beat = 1/16 bar


class Sound(NamedTuple):
    note: Note
    velocity: int
    duration: int  # duration of sound in clicks


class Track:
    """
    Track is a collection of sounds played at concrete clicks.

    This class also provides appropriate methods for working and analyzing tracks.
    """

    def __init__(self, data: Dict[int, List[Sound]]):
        self._data = data
        self._cached_intensity_matrix = None

    def to_midi_track(
        self,
        ticks_per_beat: int,
    ) -> mido.MidiTrack:
        """Convert track to MIDI track."""
        track = mido.MidiTrack()
        ticks_per_click = ticks_per_beat // 4

        to_stop: Dict[int, List[Note]] = {}
        data_clicks = sorted(self._data.keys())
        current_click = 0
        to_wait = 0

        while True:
            to_stop_clicks = sorted(to_stop.keys())

            # stop notes that should be stopped on current click
            if to_stop_clicks and to_stop_clicks[0] == current_click:
                for note in to_stop.pop(current_click):
                    track.append(
                        mido.Message(
                            "note_off",
                            note=note.midi,
                            velocity=0,
                            time=to_wait,
                        )
                    )
                    to_wait = 0

            # start notes that should be started on current click
            if data_clicks and data_clicks[0] == current_click:
                data_clicks.pop(0)
                for sound in self._data[current_click]:
                    track.append(
                        mido.Message(
                            "note_on",
                            note=sound.note.midi,
                            velocity=sound.velocity,
                            time=to_wait,
                        )
                    )
                    to_wait = 0
                    stop_at = current_click + sound.duration
                    to_stop.setdefault(stop_at, []).append(sound.note)

            # move to next click
            to_stop_clicks = sorted(to_stop.keys())
            min_to_stop = to_stop_clicks[0] if to_stop_clicks else None
            min_data = data_clicks[0] if data_clicks else None
            if min_to_stop is not None and min_data is not None:
                next_click = min(min_to_stop, min_data)
            elif min_to_stop is not None:
                next_click = min_to_stop
            elif min_data is not None:
                next_click = min_data
            else:
                break
            to_wait += (next_click - current_click) * ticks_per_click
            current_click = next_click

        return track

    @property
    def total_clicks(self) -> int:
        """Get total duration of the track in clicks."""
        last = 0
        for click, sounds in self._data.items():
            last = max(last, click + max(sound.duration for sound in sounds))
        return last

    @property
    def avg_velocity(self) -> int:
        """Get average velocity of the track."""
        total = 0
        count = 0
        for _, sounds in self._data.items():
            for sound in sounds:
                total += sound.velocity
                count += 1
        return int(round(total / count))

    def get_intensity_matrix(
        self,
        note_from: Note = Note.from_midi(21),
        note_to: Note = Note.from_midi(108),
    ) -> np.ndarray:
        """
        Get intensity matrix for the track.

        Dimension of the resulting matrix will be
        (note_to - note_from + 1)x(total track duration in ticks).

        :param note_from: Lowest note to include in the matrix.
        :param note_to: Highest note to include in the matrix.
        """
        if self._cached_intensity_matrix is not None:
            return self._cached_intensity_matrix

        from_midi = note_from.midi
        to_midi = note_to.midi

        if from_midi > to_midi:
            raise ValueError("note_from must be lower than note_to")

        intensity = {
            0: 1024,
            1: 256,
            2: 128,
            3: 64,
            4: 32,
            5: 16,
            6: 8,
            7: 4,
            8: 2,
            9: 1,
        }

        out = np.zeros((to_midi - from_midi + 1, self.total_clicks))

        for click, sounds in sorted(self._data.items()):
            for sound in filter(lambda s: from_midi <= s.note.midi <= to_midi, sounds):
                for d in range(sound.duration):
                    out[sound.note.midi - from_midi, click + d] += intensity.get(d, 0)

        self._cached_intensity_matrix = out

        return out

    def detect_key(self) -> Key:
        """Detect key of the track."""
        all_keys = {
            *MajorKey.get_all_possible_keys(),
            *MinorKey.get_all_possible_keys(),
        }

        key_hits: Dict[Key, int] = {
            key: 0 for key in all_keys
        }
        key_stability_rates: Dict[Key, List[float]] = {
            key: [0] for key in all_keys
        }
        total_notes = 0

        first_notes = None
        last_notes = None

        for _, sounds in sorted(self._data.items()):
            notes = [sound.note for sound in sounds]
            if first_notes is None:
                first_notes = notes
            last_notes = notes
            for sound in sounds:
                total_notes += 1
                note = sound.note.without_octave
                for key in all_keys:
                    if note in key.scale_set:
                        key_hits[key] += 1
                        key_stability_rates[key].append(
                            key.rate_note_stability(note)
                            * (1 + sound.velocity / 127)
                            * (1 + sound.duration / 16)
                        )

        if total_notes == 0:
            raise ValueError("track is empty")

        starting_note_rates = {
            key: max(key.rate_note_as_starting(n) for n in first_notes)
            for key in all_keys
        }
        ending_note_rates = {
            key: max(key.rate_note_as_ending(n) for n in last_notes)
            for key in all_keys
        }

        # calculate total scores for each key using all metrics
        total_scores = {
            key: (
                100*(key_hits[key] / total_notes)
                + 10*(1 + sum(key_stability_rates[key]) / len(key_stability_rates[key]))
                + 4*starting_note_rates[key]
                + 5*ending_note_rates[key]
            )
            for key in all_keys
        }

        key_score_pairs: List[Tuple[Key, float]] = list(total_scores.items())
        key_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return key_score_pairs[0][0]

    def calculate_similarity_matrix(self) -> np.ndarray:
        """Calculate similarity matrix for the track."""
        chroma_vectors = self.get_chroma_vectors()
        n = len(chroma_vectors)

        # Create a matrix of size (number of chroma vectors)x(number of chroma vectors)
        # and fill it with dot products of chroma vectors with normalized lengths.
        # Calculate only upper triangle of the matrix
        # and then copy it to the lower triangle, since the matrix is symmetric.
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if sum(chroma_vectors[i]) == 0 or sum(chroma_vectors[j]) == 0:
                    normalized_dot = 0
                else:
                    normalized_dot = np.dot(chroma_vectors[i], chroma_vectors[j]) / (
                        np.linalg.norm(chroma_vectors[i])
                        * np.linalg.norm(chroma_vectors[j])
                    )
                similarity_matrix[i, j] = normalized_dot
                similarity_matrix[j, i] = normalized_dot

        for i in range(n):
            similarity_matrix[i, i] = 0

        return similarity_matrix

    def get_chroma_vectors(self) -> List[np.ndarray]:
        """
        Get chroma vectors for the track.

        Chroma vector of 12 components that contains
        the number of notes of each pitch class encountered in
        one bar of the track.
        """
        chroma_vectors: List[np.ndarray] = []
        all_notes = tuple(Note.from_num(n+1) for n in range(12))
        for bar_notes in self.get_flat_notes_of_beats():
            counter = Counter([note.without_octave for note in bar_notes])
            chroma_vectors.append(np.array(
                [counter.get(note, 0) for note in all_notes]
            ))
        return chroma_vectors

    def get_flat_notes_of_beats(self) -> List[List[Note]]:
        """Get a list of notes contained in each beat."""
        intensity_matrix = self.get_intensity_matrix()
        total_beats = ceil(intensity_matrix.shape[1] / 4)

        # Add zeros columns to the right so matrix will have width equal total_beats*4
        intensity_matrix = np.hstack((
            intensity_matrix,
            np.zeros((intensity_matrix.shape[0], total_beats*4 - intensity_matrix.shape[1]))
        ))

        notes = []
        for i in range(total_beats):
            notes.append([
                Note.from_midi(midi + 21)
                for midi, intensity in enumerate(intensity_matrix[:, i*4:(i+1)*4].sum(axis=1))
                if intensity > 0
            ])

        return notes


class TrackBuilder:
    """TrackBuilder is a helper class for building a track."""

    def __init__(self):
        self._playing_notes: Dict[Note, List[Tuple[int, int]]] = defaultdict(list)
        self._current_click = 0
        # sounds at each click
        self._track_data: Dict[int, List[Sound]] = defaultdict(list)

    def play(self, *notes_with_velocities: Tuple[Note, int]):
        for note, velocity in notes_with_velocities:
            self._playing_notes[note].append((self._current_click, velocity))

    def wait(self, clicks: int):
        """Note: one click = 1/4 of a beat"""
        self._current_click += clicks

    def stop_playing(self, *notes: Note):
        for note in notes:
            if note in self._playing_notes:
                start_click, velocity = self._playing_notes[note].pop()
                duration = self._current_click - start_click
                self._track_data[start_click].append(Sound(note, velocity, duration))
                if len(self._playing_notes[note]) == 0:
                    del self._playing_notes[note]

    def build(self) -> Track:
        return Track(self._track_data)


class Melody:
    """
    Melody consists of melody track and optional accompaniment track.

    This class provides methods for loading and saving melodies to and from MIDI files.
    """

    def __init__(
        self,
        melody_track: Track,
        melody_track_name: Optional[str] = None,
        accompaniment_track: Optional[Track] = None,
        accompaniment_track_name: Optional[str] = None,
        time_signature: Optional[Tuple[int, int]] = None,
        tempo: Optional[int] = None,
    ):
        self._melody_track = melody_track
        self._melody_track_name = melody_track_name
        self._accompaniment_track = accompaniment_track
        self._accompaniment_track_name = accompaniment_track_name

        if time_signature is None:
            time_signature = (4, 4)
        self._time_signature = time_signature

        if tempo is None:
            self._tempo = 500000
        self._tempo = tempo

    @classmethod
    def load(cls, midi_path: str) -> "Melody":
        """Load melody from a MIDI file."""
        mid = mido.MidiFile(midi_path)
        melody_track_name = None
        time_signature = None
        tempo = None

        # build melody track
        builder = TrackBuilder()
        for track in mid.tracks:
            if track.name:
                melody_track_name = track.name
            for msg in track:
                if msg.is_meta:
                    if msg.type == "time_signature":
                        time_signature = (msg.numerator, msg.denominator)
                    elif msg.type == "set_tempo":
                        tempo = msg.tempo
                else:
                    if hasattr(msg, "time"):
                        builder.wait(round(msg.time / mid.ticks_per_beat * 4))
                    if msg.type == "note_on":
                        builder.play((Note.from_midi(msg.note), msg.velocity))
                    elif msg.type == "note_off":
                        builder.stop_playing(Note.from_midi(msg.note))

        return cls(
            builder.build(),
            melody_track_name=melody_track_name,
            time_signature=time_signature,
            tempo=tempo,
        )

    def save(self, midi_path: str):
        """Save melody to a MIDI file."""
        mid = mido.MidiFile()
        mid.ticks_per_beat = 384
        meta_track = mido.MidiTrack()
        mid.tracks.append(meta_track)

        meta_track.append(mido.MetaMessage(
            "time_signature",
            numerator=self._time_signature[0],
            denominator=self._time_signature[1],
        ))
        meta_track.append(mido.MetaMessage(
            "set_tempo",
            tempo=self._tempo,
        ))

        melody_track = self._melody_track.to_midi_track(mid.ticks_per_beat)
        if self._melody_track_name is not None:
            melody_track.name = self._melody_track_name
        mid.tracks.append(melody_track)

        if self._accompaniment_track is not None:
            accomp_track = self._accompaniment_track.to_midi_track(mid.ticks_per_beat)
            if self._accompaniment_track_name is not None:
                accomp_track.name = self._accompaniment_track_name
            mid.tracks.append(accomp_track)

        mid.save(midi_path)

    @property
    def melody_track(self) -> Track:
        return self._melody_track

    @property
    def accompaniment_track(self) -> Optional[Track]:
        return self._accompaniment_track


# Distributions of all types of chords used to generate random chords
CHORDS_DISTRIBUTION = {
    ChordType.REST: 0.07,

    # Diminished + suspended = 0.13
    ChordType.DIM: 0.01,
    ChordType.SUS2_1: 0.01,
    ChordType.SUS2_2: 0.01,
    ChordType.SUS2_3: 0.01,
    ChordType.SUS2_4: 0.01,
    ChordType.SUS2_5: 0.01,
    ChordType.SUS4_1: 0.01,
    ChordType.SUS4_2: 0.01,
    ChordType.SUS4_3: 0.01,
    ChordType.SUS4_4: 0.01,
    ChordType.SUS4_5: 0.01,

    # Major chords = 0.40
    ChordType.MAJOR1: 0.08,
    ChordType.MAJOR1_INV1: 0.03,
    ChordType.MAJOR1_INV2: 0.03,
    ChordType.MAJOR2: 0.08,
    ChordType.MAJOR2_INV1: 0.03,
    ChordType.MAJOR2_INV2: 0.03,
    ChordType.MAJOR3: 0.07,
    ChordType.MAJOR3_INV1: 0.03,
    ChordType.MAJOR3_INV2: 0.03,

    # Minor chords = 0.40
    ChordType.MINOR1: 0.08,
    ChordType.MINOR1_INV1: 0.03,
    ChordType.MINOR1_INV2: 0.03,
    ChordType.MINOR2: 0.08,
    ChordType.MINOR2_INV1: 0.03,
    ChordType.MINOR2_INV2: 0.03,
    ChordType.MINOR3: 0.07,
    ChordType.MINOR3_INV1: 0.03,
    ChordType.MINOR3_INV2: 0.03,
}
CHORDS_DISTRIBUTION_CHORDS = list(CHORDS_DISTRIBUTION.keys())
CHORDS_DISTRIBUTION_PROBS = list(CHORDS_DISTRIBUTION.values())
# check that distribution sums to 1
assert abs(sum(CHORDS_DISTRIBUTION_PROBS) - 1.0) < 1e-6


class Gene:
    """
    Gene represents a single chord played in specific octave.
    """
    chord: ChordType
    octave: int

    def __init__(self, chord: ChordType, octave: int):
        self.chord = chord
        self.octave = octave

    @classmethod
    def random(cls) -> "Gene":
        """Generate a random gene (random chord in random octave)."""
        chord = np.random.choice(
            CHORDS_DISTRIBUTION_CHORDS,
            p=CHORDS_DISTRIBUTION_PROBS,
        )
        octave = np.random.randint(2, 5+1)
        return cls(chord, octave)


class Chromosome:
    """
    Chromosome represents a list of genes (or chords) that is
    it can be converted to an accompanying track.
    """
    genes: List[Gene]

    def __init__(self, genes: List[Gene]):
        self.genes = genes

    @classmethod
    def random(cls, length: int) -> "Chromosome":
        """Generate a random chromosome of given length."""
        return cls([Gene.random() for _ in range(length)])

    def to_track(self, key: Key, velocity: int) -> Track:
        """Convert chromosome to a track."""
        builder = TrackBuilder()
        for gene in self.genes:
            if gene.chord is ChordType.REST:
                builder.wait(4*2)  # wait 2 bars
            else:
                notes = key.get_chord_notes(gene.chord, gene.octave)
                builder.play(*[(note, velocity) for note in notes])
                builder.wait(4*2)  # wait 2 bars
                builder.stop_playing(*notes)
        return builder.build()

    def mutate(self, mutation_rate: float):
        """Mutate the chromosome."""
        for i in range(len(self.genes)):
            if np.random.random() < mutation_rate:
                self.genes[i] = Gene.random()

    @classmethod
    def crossover(cls, parent1: "Chromosome", parent2: "Chromosome") -> "Chromosome":
        """Crossover two chromosomes."""

        # select a random crossover point
        crossover_point = np.random.randint(0, len(parent1.genes))
        # create a new chromosome with genes from both parents
        genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]

        assert len(genes) == len(parent1.genes) and len(genes) == len(parent2.genes)

        return cls(genes)


class Generation:
    """
    Generation represents a population of chromosomes (accompaniments).

    It is used to generate a new generation of chromosomes and select the best ones.
    """
    def __init__(self, chromosomes: List[Chromosome]):
        self._chromosomes = chromosomes
        self._scores = None
        self._scores_dist = None

    def _update_scores(
        self,
        fitness_function: Callable[[Chromosome], float],
    ):
        self._scores = np.array([fitness_function(chromosome) for chromosome in self._chromosomes])
        self._scores_dist = self._scores / np.sum(self._scores)

    @property
    def size(self) -> int:
        return len(self._chromosomes)

    @classmethod
    def random(
        cls,
        size: int,
        chromosome_length: int,
    ) -> "Generation":
        """
        Generate a random generation of chromosomes.

        :param size: Number of chromosomes in the generation.
        :param chromosome_length: Length of each chromosome.
        """
        return cls([Chromosome.random(chromosome_length) for _ in range(size)])

    def get_best_chromosome(
        self,
        fitness_function: Callable[[Chromosome], float] = None,
    ) -> Tuple[Chromosome, float]:
        """Return the best chromosome and its score."""
        if fitness_function is not None:
            self._update_scores(fitness_function)
        elif self._scores is None:
            raise ValueError("no fitness function provided and scores not computed yet")
        best_idx = np.argmax(self._scores)
        return self._chromosomes[best_idx], self._scores[best_idx]

    def get_next_generation(
        self,
        fitness_function: Callable[[Chromosome], float],
        keep_best_ratio: float,
        mutate_ratio: float,
        mutation_rate: float,
    ) -> "Generation":
        """
        Perform selection, crossover and mutation to generate a new generation,
        and return it.
        """
        keep_best_size = int(round(self.size * keep_best_ratio))
        to_produce_size = self.size - keep_best_size

        if self._scores is None:
            self._update_scores(fitness_function)

        # crossover
        new_chromosomes = []
        indexes = tuple(range(len(self._chromosomes)))
        for _ in range(to_produce_size):
            i1, i2 = np.random.choice(
                indexes,
                size=2,
                replace=False,
                p=self._scores_dist,
            )
            p1 = self._chromosomes[i1]
            p2 = self._chromosomes[i2]
            new_chromosomes.append(Chromosome.crossover(p1, p2))

        # add the best chromosomes (elitism)
        if keep_best_size > 0:
            best_indexes = np.argpartition(self._scores, -keep_best_size)[-keep_best_size:]
            new_chromosomes.extend([self._chromosomes[i] for i in best_indexes])

        # mutate
        for chromosome in new_chromosomes:
            if np.random.random() < mutate_ratio:
                chromosome.mutate(mutation_rate)

        new = Generation(new_chromosomes)
        assert new.size == self.size

        return new


def generate_accompaniment_ga(
    melody: Melody,
    key: Key,
    generation_size: int,
    generations_count: int,
    on_percent_done: Callable[[int], None] = None,
) -> Track:
    """
    Generate accompaniment using genetic algorithm.

    :param melody: Melody to accompany.
    :param key: Key of the melody.
    :param generation_size: Number of chromosomes in each generation.
    :param generations_count: Number of generations to generate.
    :param on_percent_done: Callback to call on each percent of progress.
    """

    melody_track = melody.melody_track
    velocity = melody_track.avg_velocity

    # reduce the accompaniment velocity a bit
    velocity = round(velocity * 0.83)

    bars_count = ceil(melody_track.total_clicks / 4)
    chromosome_length = bars_count // 2  # 2 bars per chord (gene)

    # generate initial random population
    population = Generation.random(
        size=generation_size,
        chromosome_length=chromosome_length,
    )

    melody_similarity_matrix = melody_track.calculate_similarity_matrix()

    # define fitness function here to extract common logic
    # and also avoid recomputing the similarity matrix
    def fitness_function(chromosome: Chromosome) -> float:
        # generate accompaniment track from chromosome (chords)
        accomp_track = chromosome.to_track(key, velocity)

        # rate accompaniment similarity patterns
        accomp_similarity_matrix = accomp_track.calculate_similarity_matrix()
        similarity_score = get_square_matrices_similarity_score(
            melody_similarity_matrix,
            accomp_similarity_matrix,
        )

        dissonance_penalty = get_dissonance_penalty(melody_track, accomp_track)
        note_distances_penalty = get_note_distances_penalty(melody_track, accomp_track)
        total_silence_penalty = get_silence_penalty(melody_track, accomp_track)
        accompaniment_silence_penalty = get_silence_penalty(accomp_track)

        return (
            similarity_score * 100
            * (1 - dissonance_penalty)
            * (1 - note_distances_penalty)**5
            * (1 - total_silence_penalty)
            * (1 - accompaniment_silence_penalty)**2
        )

    best, best_score = population.get_best_chromosome(fitness_function)
    for i in range(generations_count):
        population = population.get_next_generation(
            fitness_function,
            keep_best_ratio=0.05,  # keep 5% of best chromosomes
            mutate_ratio=0.8,  # mutate 80% of chromosomes
            mutation_rate=0.1,  # mutate 10% of genes
        )
        new_best, new_best_score = population.get_best_chromosome(fitness_function)
        if new_best_score > best_score:
            best, best_score = new_best, new_best_score

        if on_percent_done is not None:
            on_percent_done(round((i + 1) / generations_count * 100))

    return best.to_track(key, velocity)


def get_square_matrices_similarity_score(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
) -> float:
    """
    Return the similarity score of two square matrices.

    Two matrices are considered similar if they have the same
    "white" patterns in the same places.
    """
    m1_w, m1_h = matrix1.shape
    m2_w, m2_h = matrix2.shape

    # matrices must be square
    assert m1_w == m1_h, "matrix 1 is not square"
    assert m2_w == m2_h, "matrix 2 is not square"

    # make them the same size
    size = max(m1_w, m2_w)
    matrix1 = np.pad(
        matrix1,
        ((0, size-m1_w), (0, size-m1_h)),
        "constant",
    )
    matrix2 = np.pad(
        matrix2,
        ((0, size-m2_w), (0, size-m2_h)),
        "constant",
    )

    # square errors
    error = 0
    counted = 0
    for i in range(size):
        for j in range(i, size):
            if matrix1[i, j] != 0 and matrix2[i, j] != 0:
                error += (matrix1[i, j] - matrix2[i, j]) ** 2
                counted += 1

    if counted == 0:
        return 0

    mean = error / counted

    return 1.0 / (1.0 + mean)


def get_dissonance_penalty(*tracks: Track) -> float:
    """
    Get dissonance penalty for the tracks.

    :param tracks: Tracks to get dissonance penalty for.
    :return: Dissonance penalty.
    """

    penalty = 0
    max_penalty = 0

    all_notes_by_beats = defaultdict(list)

    for track in tracks:
        for i, notes in enumerate(track.get_flat_notes_of_beats()):
            all_notes_by_beats[i].extend(notes)

    for notes in all_notes_by_beats.values():
        penalty += get_dissonance_penalty_for_notes(notes)
        max_penalty += 1.0

    if max_penalty == 0:
        return 0

    return penalty / max_penalty


def get_dissonance_penalty_for_notes(notes: List[Note]) -> float:
    """
    Get dissonance penalty for the notes.

    :param notes: Notes to get dissonance penalty for.
    :return: Dissonance penalty.
    """

    penalty = 0

    if len(notes) == 1:
        return penalty

    max_penalty = 0
    for i in range(len(notes)):
        for j in range(i+1, len(notes)):
            midi_a, midi_b = notes[i].midi, notes[j].midi
            max_penalty += 2
            if abs(midi_a - midi_b) == 1:
                penalty += 2
            elif abs(midi_a - midi_b) == 2:
                penalty += 1
            elif abs(midi_a - midi_b) == 8:
                penalty += 1

    if max_penalty == 0:
        return 0

    return penalty / max_penalty


def get_note_distances_penalty(track1: Track, track2: Track) -> float:
    """
    Get note distances penalty for the tracks.

    Penalty higher when "center of masses" of notes
    are far away from each other.

    :param track1: First track.
    :param track2: Second track.
    :return: Note distances penalty.
    """

    penalty = 0

    notes_of_beats1 = track1.get_flat_notes_of_beats()
    notes_of_beats2 = track2.get_flat_notes_of_beats()

    # make them the same size
    size = max(len(notes_of_beats1), len(notes_of_beats2))
    if len(notes_of_beats1) < size:
        notes_of_beats1.extend([[]] * (size - len(notes_of_beats1)))
    elif len(notes_of_beats2) < size:
        notes_of_beats2.extend([[]] * (size - len(notes_of_beats2)))

    prev_notes1 = []
    prev_notes2 = []

    max_total_distances = 0

    for beat_notes1, beat_notes2 in zip(
        notes_of_beats1,
        notes_of_beats2,
    ):
        if not beat_notes1 and not beat_notes2:
            continue

        skip = False
        if not beat_notes1 and beat_notes2 and prev_notes1:
            center_of_mass1 = sum(n.midi for n in prev_notes1) / len(prev_notes1)
            center_of_mass2 = sum(n.midi for n in beat_notes2) / len(beat_notes2)
        elif beat_notes1 and not beat_notes2 and prev_notes2:
            center_of_mass1 = sum(n.midi for n in beat_notes1) / len(beat_notes1)
            center_of_mass2 = sum(n.midi for n in prev_notes2) / len(prev_notes2)
        elif beat_notes1 and beat_notes2:
            center_of_mass1 = sum(n.midi for n in beat_notes1) / len(beat_notes1)
            center_of_mass2 = sum(n.midi for n in beat_notes2) / len(beat_notes2)
        else:
            skip = True
            center_of_mass1 = 0
            center_of_mass2 = 0

        if not skip:
            distance = abs(center_of_mass1 - center_of_mass2)
            # distance should not be too small and not too big
            if distance < 10:
                penalty += 10 - distance
                max_total_distances += 10
            elif distance > 14:
                penalty += (distance - 14) * 2
                max_total_distances += penalty

        prev_notes1 = beat_notes1 or prev_notes1
        prev_notes2 = beat_notes2 or prev_notes2

    return penalty / max_total_distances


def get_silence_penalty(*tracks: Track) -> float:
    """
    Get silence penalty for the tracks.

    Note: assume they played together.

    :param tracks: Tracks used to measure silence penalty.
    :return: Silence penalty.
    """

    assert len(tracks) > 0

    intensity_matrices = [
        track.get_intensity_matrix()
        for track in tracks
    ]

    equal = []
    # make all matrices the same size
    size = max(m.shape[1] for m in intensity_matrices)
    for m in intensity_matrices:
        if m.shape[1] < size:
            m = np.hstack((m, np.zeros((m.shape[0], size - m.shape[1]))))
        equal.append(m)

    max_penalty = 0
    penalty = 0
    counter = 0
    for i in range(equal[0].shape[1]):
        # get the average intensity of all tracks
        # at this point in time
        sum_of_intensities = 0
        for m in equal:
            sum_of_intensities += m[:, i].sum()

        max_penalty += 1

        if sum_of_intensities > 0:
            counter = 4
        else:
            if counter <= 0:
                penalty += 1
            else:
                counter -= 1

    if max_penalty == 0:
        return 0

    return penalty / max_penalty


def detect_key_krumhansl_schmuckler(path: str) -> Key:
    """
    Detect key of a melody from MIDI file using
    music21's implementation of Krumhansl-Schmuckler algorithm.

    :param path: path to MIDI file with melody to detect key of.
    """
    from music21 import converter

    stream = converter.parse(path)
    key = stream.analyze("key")

    key_21, mode_21 = str(key.tonic), str(key.mode)
    if key_21 and mode_21:
        mode_21 = mode_21.lower()
        if mode_21 == "major":
            return MajorKey(Note(key_21))
        elif mode_21.lower() == "minor":
            return MinorKey(Note(key_21))

    raise ValueError("music21 failed to detect key")


def parse_melody_with_key_from_midi(path: str) -> Tuple[Melody, Key]:
    melody = Melody.load(path)

    try:
        key = detect_key_krumhansl_schmuckler(path)
    except BaseException:
        logger.debug("Error", exc_info=True)
        logger.warning("music21 failed to analyze melody key, using built-in algorithm")
        key = melody.melody_track.detect_key()

    return melody, key


def export_numpy_array_to_image(arr: np.ndarray, path: str) -> None:
    """
    Export numpy array to grayscale image.

    :param arr: numpy array to export (should have integer values in range [0, 255]).
    :param path: path to save image to.
    """
    try:
        import PIL.Image
    except ImportError:
        logger.warning("failed to import Pillow (required to export matrix to image)")
        return

    arr = arr.astype(np.uint8)
    img = PIL.Image.fromarray(arr)
    img.save(path)


def parse_args() -> argparse.Namespace:
    """Define and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate accompaniment for monophonic melodies."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=str,
        help="paths to source MIDI files of monophonic melodies",
    )
    parser.add_argument(
        "-m", "--similarity-matrix",
        default=False,
        help="whether to export similarity matrix for each melody (Pillow is required)",
        action="store_true",
    )
    parser.add_argument(
        "-g", "--generation-size",
        type=int,
        default=300,
        help="size of each generation (>=10)",
    )
    parser.add_argument(
        "-n", "--generations-count",
        type=int,
        default=20,
        help="total number of generations (>=1)",
    )
    parser.add_argument(
        "-v", "--verbose",
        default=False,
        help="print debug info",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Validate arguments
    gen_size = args.generation_size
    if gen_size < 10:
        print("Error: minimum generation size is 10, please use a larger value")
        return
    gen_count = args.generations_count
    if gen_count < 1:
        print("Error: minimum number of generations is 1, please use a larger value")
        return
    should_export_matrix = args.similarity_matrix

    inputs: List[Tuple[Melody, Key, str]] = []

    print("### Parsing melodies ###")
    for i, source in enumerate(args.sources):
        print(f"#  {i+1}. {source}", end="")
        melody, key = parse_melody_with_key_from_midi(source)

        if source.endswith(".mid"):
            source = source[:-4]

        print(f" ✔ '{key}' (detected key)")

        inputs.append((melody, key, source))
    print("########################\n")
    # end of parsing

    # generate accompaniment for each melody
    print(f"### Generating accompaniments ###")
    print(f"#--- {gen_count} generations of size {gen_size} for each melody")
    for i, (melody, key, filename) in enumerate(inputs):
        print(f"# {i+1}. {filename}")

        next_percent_step = 10

        def on_percent_done(percent: float) -> None:
            nonlocal next_percent_step
            if percent >= next_percent_step:
                print(f"#", end="", flush=True)
                next_percent_step += 10

        print(f"#    > Generating accompaniment [", end="", flush=True)
        accomp = generate_accompaniment_ga(
            melody,
            key,
            generation_size=gen_size,
            generations_count=gen_count,
            on_percent_done=on_percent_done,
        )
        melody._accompaniment_track = accomp
        print("] ✔")

        output_path = f"VladislavDeryabkinOutput{i+1}-{key.short_name}.mid"
        print(f"#    > Saving to {output_path}", end="")
        melody.save(output_path)
        print(" ✔")

        if should_export_matrix:
            print("#    > Exporting similarity matrices", end="")

            melody_mat_path = filename + "__melody.png"
            accomp_mat_path = filename + "__accomp.png"

            melody_mat = melody.melody_track.calculate_similarity_matrix()
            accomp_mat = accomp.calculate_similarity_matrix()

            try:
                export_numpy_array_to_image(melody_mat * 255, melody_mat_path)
                export_numpy_array_to_image(accomp_mat * 255, accomp_mat_path)
            except BaseException:
                logger.debug("Error", exc_info=True)
                print(" ❌ (failed to export matrices)")
            else:
                print(" ✔")
    print("#################################\n")
    print("Finished.")


if __name__ == "__main__":
    main()
