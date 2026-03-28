class SignToText:

    def __init__(self):
        self.buffer = []

    def update(self, gesture):

        if gesture:
            self.buffer.append(gesture)

        if len(self.buffer) > 5:
            sentence = " ".join(self.buffer)
            self.buffer.clear()
            return sentence

        return None
