"""
Gesture Mapper — Maps sign language tokens to animation sequences.

Converts recognized gestures/gloss tokens into animation clip IDs
that drive the 3D avatar.
"""


class GestureMapper:

    def __init__(self):
        # Map sign tokens to animation clip names
        self.animation_map = {
            "HELLO": "hello_wave",
            "THANK_YOU": "thank_you_nod",
            "YES": "yes_nod",
            "NO": "no_shake",
            "I": "point_self",
            "YOU": "point_forward",
            "GO": "walk_gesture",
            "SCHOOL": "school_sign",
            "EAT": "eat_gesture",
            "DRINK": "drink_gesture",
            "HELP": "help_sign",
            "PLEASE": "please_sign",
            "SORRY": "sorry_sign",
            "LOVE": "love_sign",
            "FRIEND": "friend_sign",
        }

        # Fingerspelling for unknown words
        self.alphabet_map = {
            chr(c): f"finger_{chr(c)}" for c in range(ord('A'), ord('Z') + 1)
        }

    def map_token(self, token):
        """Map a single sign token to an animation clip name."""
        if token in self.animation_map:
            return self.animation_map[token]
        # Fallback: fingerspell the word
        return [self.alphabet_map.get(c, "finger_unknown") for c in token]

    def map_sequence(self, tokens):
        """Map a list of sign tokens to animation sequence."""
        sequence = []
        for token in tokens:
            result = self.map_token(token)
            if isinstance(result, list):
                sequence.extend(result)
            else:
                sequence.append(result)
        return sequence
