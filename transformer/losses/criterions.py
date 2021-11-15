import numpy as np
import torch
from torch import nn
from transformer.assertions.interface import Assertion
from transformer.losses.custom_loss import UnlikelihoodLoss2d
from transformer.losses.utils import calculate_bleu

class CriterionInterface(Assertion):
    criterion = None

    def set_device(self, device):
        self.criterion = self.criterion.to(device)

    def get_loss(self, predictions, targets):
        self.assert_implemented(method_name="get_loss")

    def get_accuracy(self, predictions, targets):
        self.assert_implemented(method_name="get_accuracy")

class LanguageModelingCriterion(CriterionInterface):
    def __init__(self, ignore_index, ngram, clipping, is_log_prob=True):
        self.ignore_index = ignore_index
        self.ngram = ngram
        self.bleu_weights = None
        self.clipping = clipping
        self.is_log_prob = is_log_prob
        self.criterion = nn.NLLLoss(ignore_index=ignore_index) # NLLLoss(log(softmax(x)) == CrossEntropyLoss(x)

    def get_loss(self, predictions, targets):
        '''
        :param predictions: (batch_size, timesteps, vocab_size)
        :param targets: (batch_size, timesteps)
        :return:
        '''
        # predictions: (batch_size * timesteps, vocab_size)
        predictions = predictions.view(-1, predictions.size(-1))
        # targets: (batch_size * timesteps, )
        targets = targets.contiguous().view(-1)

        loss = self.criterion(predictions, targets)
        return loss

    def get_accuracy(self, predictions, targets):
        '''
        :param predictions: (batch_size, timesteps, vocab_size)
        :param targets: (batch_size, timesteps)
        :return:
        '''
        # predictions: (batch_size * timesteps, vocab_size)
        predictions = predictions.view(-1, predictions.size(-1))
        # predictions: (batch_size * timesteps, )
        predictions = torch.argmax(predictions, axis=-1)
        # targets: (batch_size * timesteps, )
        targets = targets.contiguous().view(-1)

        label_weights = None
        if self.ignore_index is not None: label_weights = (targets != self.ignore_index).float()
        else: label_weights = torch.ones_like(targets)

        numerator = torch.sum(label_weights * (predictions == targets).float())
        denominator = torch.sum(label_weights) + 1e-5
        accuracy = numerator / denominator
        return accuracy

    def get_perplexity(self, predictions: torch.Tensor, targets: torch.Tensor):
        '''
        :param predictions: (batch_size, timesteps, vocab_size)
        :param targets: (batch_size, timesteps)
        :return:
        '''
        # predictions: (batch_size * timesteps, vocab_size)
        batch_size = predictions.shape[0]
        predictions = predictions.view(-1, predictions.size(-1))
        # targets: (batch_size * timesteps, )
        targets = targets.contiguous().view(-1)

        label_weights = None
        if self.ignore_index is not None: label_weights = (targets != self.ignore_index).float()
        else: label_weights = torch.ones_like(targets)
        n = torch.sum(label_weights.reshape(batch_size, -1), axis=-1)

        p_w = predictions.gather(1, targets.unsqueeze(1)).squeeze()
        p_w = p_w * label_weights
        p_w = torch.sum(p_w.reshape(batch_size, -1), axis=-1)
        batch_perplexities = -1 / n * p_w
        if self.is_log_prob: batch_perplexities = torch.exp(batch_perplexities)
        perplexity = torch.mean(batch_perplexities)
        return perplexity

    def get_bleu_score(self, predictions: torch.Tensor, targets: torch.Tensor):
        '''
        :param predictions: (batch_size, timesteps, vocab_size)
        :param targets: (batch_size, timesteps)
        :return:
        '''
        pad_token_id = eos_token_id = -1
        if self.ignore_index is not None: pad_token_id = self.ignore_index
        # predictions, targets: (batch_size, timesteps)
        predictions = predictions.cpu().detach().numpy()
        predictions = np.argmax(predictions, axis=-1).tolist()
        targets = targets.cpu().detach().numpy().tolist()

        if pad_token_id in targets[0]: eos_token_id = targets[0][targets[0].index(pad_token_id)-1]
        else: eos_token_id = targets[0][-1]
        bleu_score = calculate_bleu(predictions=predictions, targets=targets, ngram=self.ngram, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
        return bleu_score

class LogLikelyhoodCriterion(CriterionInterface):
    def __init__(self):
        self.criterion = nn.NLLLoss() # NLLLoss(log(softmax(x)) == CrossEntropyLoss(x)

    def get_loss(self, predictions, targets):
        '''
        :param predictions: (batch_size, num_class)
        :param targets: (batch_size, )
        :return:
        '''
        # predictions: (batch_size, num_class)
        # targets: (batch_size, )
        loss = self.criterion(predictions, targets)
        return loss

    def get_accuracy(self, predictions, targets):
        '''
        :param predictions: (batch_size, num_class)
        :param targets: (batch_size, )
        :return:
        '''
        # predictions: (batch_size, num_class)
        # targets: (batch_size, )
        # predictions: (batch_size, )
        predictions = torch.argmax(predictions, axis=-1)
        label_weights = torch.ones_like(targets)
        numerator = torch.sum((predictions == targets).float())
        denominator = torch.sum(label_weights) + 1e-5
        accuracy = numerator / denominator
        return accuracy

class UnlikelihoodCriterion(CriterionInterface):
    def __init__(self, timesteps, vocab_size, ignore_index, ngram: int = 5, is_log_probs: bool = True):
        self.timesteps = timesteps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.ngram = ngram
        self.is_log_probs = is_log_probs
        self.criterion = UnlikelihoodLoss2d(timesteps=timesteps, vocab_size=vocab_size, ignore_index=ignore_index, ngram=ngram, is_log_probs=is_log_probs)

    def get_loss(self, predictions, targets):
        '''
        :param predictions: (batch_size, timesteps, vocab_size)
        :param targets: (batch_size, timesteps)
        :return:
        '''
        loss = self.criterion(predictions, targets)
        return loss

    def get_accuracy(self, predictions, targets):
        '''
        :param predictions: (batch_size, timesteps, vocab_size)
        :param targets: (batch_size, timesteps)
        :return:
        '''
        # predictions: (batch_size * timesteps, vocab_size)
        predictions = predictions.view(-1, predictions.size(-1))
        # predictions: (batch_size * timesteps, )
        predictions = torch.argmax(predictions, axis=-1)
        # targets: (batch_size * timesteps, )
        targets = targets.contiguous().view(-1)

        numerator = 0
        denominator = (targets != self.ignore_index).float().sum() + 1e-5
        for idx in range(0, len(predictions)):
            if targets[idx] == self.ignore_index: continue
            if idx % self.timesteps != 0:
                begin_idx = idx - idx % self.timesteps
                if predictions[idx] in targets[begin_idx:idx]: continue
            numerator += 1
        accuracy = numerator / denominator
        return accuracy