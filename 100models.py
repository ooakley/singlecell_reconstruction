"""
100 models in three days. I guess we'll see how it goes lol.

Example:
python 100models.py
"""


class section_classifier(nn.Module):
    def __init__(self, section_id):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=False)
        self.lin1 = nn.Linear(numgene, 1500)
        self.lin2 = nn.Linear(1500, 100)
        self.sm = nn.Softmax(1)

def main():
