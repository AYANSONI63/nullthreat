import torch 
import torch.nn as nn


class URLClassifier(nn.Module):

    def __init__(
       self,
       vocab_size,
       embedding_dim,
       hidden_size,
       num_layers,
       num_classes,
       dropout
    ):
        
        super().__init__()

        # Adding the Embedding layer 

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0 
        )
        
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size * 2,
            num_classes
        )
        
    
    def forward(self, x):

        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        hidden = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        hidden = self.dropout(hidden)

        logits = self.fc(hidden)


        return logits

        