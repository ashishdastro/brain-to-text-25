import torch
from torch import nn


class RecurrentModel(nn.Module):
    """
    Brain-to-text RNN baseline model.

    Pipeline:
        [B, T, 512] -> Linear(512→256) -> RNN -> Linear -> LogSoftmax(CTC)
    """

    def __init__(self,
                 model_type,
                 data_input_size,
                 adapter_output_size,
                 hidden_size,
                 output_size,          # <-- add this
                 num_layers,
                 bidirectional):
        super().__init__()

        self.model_type = model_type.upper()
        self.bidirectional = bidirectional

        # 1️⃣ Adapter layer: project neural features down
        self.adapter = nn.Linear(512, 256)

        # 2️⃣ Recurrent backbone
        rnn_args = dict(
            input_size=adapter_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if self.model_type == "RNN":
            self.rnn = nn.RNN(**rnn_args)
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(**rnn_args)
        elif self.model_type == "GRU":
            self.rnn = nn.GRU(**rnn_args)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # 3️⃣ Classifier head
        fc_in = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, output_size)

    def forward(self, x):
        """
        x : [B, T, 512]
        returns log_probs : [B, T, num_classes]
        """

        # Adapter
        x = self.adapter(x)

        # RNN
        out, _ = self.rnn(x)

        # Class logits
        out = self.fc(out)

        # CTC requires log-probs
        return torch.log_softmax(out, dim=-1)
