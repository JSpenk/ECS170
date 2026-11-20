'''
Base MethodModule class for all models and frameworks
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD


from local_code.stage_4_code.Method_RNN_Classification import Method_RNN

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Method_RNN_Generation(Method_RNN):

    def __init__(self, mName, mDescription, **kwargs):
        kwargs['bidirectional'] = False

        super().__init__(mName, mDescription, **kwargs)

        self.classifier = nn.Linear(
            self.rnn_hidden,
            self.vocab_size
        )

        self.to(self.device)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.classifier(output)

    def _build_vocabulary(self, texts):
        super()._build_vocabulary(texts)

        self.word_to_id['<EOS>'] = len(self.word_to_id)

    def _prepare_generation_data(self, texts):
        sequences = []

        for text in texts:
            tokens = self._tokenize(text)

            # words --> vocabulary IDs
            ids = [
                self.word_to_id.get(token, 1)
                for token in tokens
            ]
            ids.append(self.word_to_id['<EOS>'])

            if len(ids) < 2:
                continue

            # input
            x = ids[:-1]

            # target
            y = ids[1:]

            sequences.append((x, y))

        return sequences

    def _train_generation(self, sequences):
        sequence_length = min(
            max(len(x) for x, y in sequences),
            self.max_length
        )

        padded_x = []
        padded_y = []

        for x, y in sequences:
            x = x[:sequence_length]
            y = y[:sequence_length]

            x = x + [0] * (sequence_length - len(x))
            y = y + [0] * (sequence_length - len(y))

            padded_x.append(x)
            padded_y.append(y)

        input_tensor = torch.tensor(padded_x, dtype=torch.long)
        target_tensor = torch.tensor(padded_y, dtype=torch.long)

        dataset = TensorDataset(input_tensor, target_tensor)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = self._make_optimizer()

        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'train_perplexity': [],
            'epoch': [],
        }

        for epoch in range(self.max_epoch):

            self._set_mode(True)

            total_loss = 0.0
            total_correct = 0
            total_tokens = 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.forward(batch_x)

                loss = criterion(
                    predictions.reshape(-1, self.vocab_size),
                    batch_y.reshape(-1)
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=1.0
                )

                optimizer.step()
                mask = batch_y != 0
                predicted_tokens = predictions.argmax(dim=-1)

                correct = (
                        (predicted_tokens == batch_y) & mask
                ).sum().item()

                token_count = mask.sum().item()
                total_correct += correct
                total_tokens += token_count
                total_loss += loss.item() * token_count

            # epoch metrics
            average_loss = total_loss / total_tokens
            accuracy = total_correct / total_tokens
            perplexity = torch.exp(
                torch.tensor(average_loss)
            ).item()

            self.history['train_loss'].append(average_loss)
            self.history['train_accuracy'].append(accuracy)
            self.history['train_perplexity'].append(perplexity)
            self.history['epoch'].append(epoch + 1)

            print(
                "Epoch:", epoch + 1,
                "Loss:", round(average_loss, 4),
                "Accuracy:", round(accuracy, 4),
                "Perplexity:", round(perplexity, 4)
            )

    def run(self):
        texts = self.data['X']

        self._build_vocabulary(texts)

        self.vocab_size = len(self.word_to_id)

        self.classifier = nn.Linear(
            self.rnn_hidden,
            self.vocab_size
        ).to(self.device)

        sequences = self._prepare_generation_data(texts)

        print("Vocabulary size:", self.vocab_size)
        print("Number of training sequences:", len(sequences))

        self._train_generation(sequences)

        starting_phrases = [
            "Why did the",
            "What is the",
            "Why are there",
            "What do you"
        ]

        print("\n===== Generation Evaluation =====")

        print("Final Loss:",
              round(self.history['train_loss'][-1], 4))

        print("Final Token Accuracy:",
              round(self.history['train_accuracy'][-1] * 100, 2), "%")

        print("Final Perplexity:",
              round(self.history['train_perplexity'][-1], 4))

        for phrase in starting_phrases:
            generated = self.generate_text(phrase, num_words=20)

            print("\nStarting phrase:", phrase)
            print("Generated text:", generated)

        return sequences

    def generate_text(self, start_text, num_words=20):

        self._set_mode(False)

        tokens = self._tokenize(start_text)

        # words --> vocabulary IDs
        ids = [
            self.word_to_id.get(token, 1)
            for token in tokens
        ]

        # Reverse
        id_to_word = {
            index: word
            for word, index in self.word_to_id.items()
        }

        # Generate words
        for _ in range(num_words):
            input_tensor = torch.tensor(
                [ids],
                dtype=torch.long
            ).to(self.device)

            with torch.no_grad():
                predictions = self.forward(input_tensor)

            next_word_logits = predictions[0, -1, :].clone()

            next_word_logits[0] = float('-inf')
            next_word_logits[1] = float('-inf')

            temperature = 0.7
            next_word_logits = next_word_logits / temperature

            top_values, top_indices = torch.topk(
                next_word_logits,
                k=5
            )

            probabilities = torch.softmax(top_values, dim=0)

            selected_index = torch.multinomial(
                probabilities,
                num_samples=1
            ).item()

            next_word_id = top_indices[selected_index].item()

            if next_word_id == self.word_to_id['<EOS>']:
                break

            ids.append(next_word_id)

        # IDs back into words
        generated_words = [
            id_to_word.get(index, '<UNK>')
            for index in ids
        ]

        return ' '.join(generated_words)
